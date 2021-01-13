import cv2
import numpy as np
from imutils import contours

def find_squares(img):
    """
    Method to detect the squares in the image
    :param img: image of cube
    :type img: np.ndarray
    :return: mask that identifies the squares in image, and sorted contours
    :rtype:  tuple[np.ndarray, list]"""

    # Boundary of colors to detect
    colorBoundaries = {
        'gray': ([76, 0, 41], [179, 255, 70]),  # Gray
        'blue': ([69, 120, 100], [179, 255, 255]),  # Blue
        'yellow': ([21, 110, 117], [45, 255, 255]),  # Yellow
        'orange': ([0, 110, 125], [17, 255, 255]),  # Orange
        'red': ([17, 15, 100], [50, 56, 200]) # Red
    }

    # Create mask
    mask = np.zeros(img.shape, dtype=np.uint8)

    # Create kernel for morphological operations
    square_open_kernel = np.ones((7, 7), np.uint8)
    square_close_kernel = np.ones((5, 5),np.uint8)
    # Loop through the boundaries
    for (lower, upper) in colorBoundaries.values():
        # Create numpy arrays from the boundaries
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)

        # find the colors within the specified boundaries and apply
        # the mask
        color_mask = cv2.inRange(img, lower, upper)
        image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        # Perform open and close morphology operations to remove noise
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, square_open_kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, square_close_kernel, iterations=5)

        color_mask = cv2.merge([color_mask, color_mask, color_mask])
        mask = cv2.bitwise_or(mask, color_mask)

    # Convert mask to grayscale
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Find contours in grayscale mask
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Account for OpenCV3 since cv2.findContours in OpenCV3 returns 3 values in the following order (image, contours, hierarchy)
    # Open CV version < 3 returns two values (contours, hierarchy)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Sort all contours from top-to-bottom
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    # Take each row of 3 and sort contours from left-to-right
    cube_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        row.append(c)
        if i % 3 == 0:
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            cube_rows.append(cnts)
            row = []

    return mask, cube_rows

if __name__ == '__main__':
    img = cv2.imread('2.png')
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask, cube_contours = find_squares(hsvImg)





