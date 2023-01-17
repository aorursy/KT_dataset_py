# This is the base algorithm and it supposed to be forked for additional improvements.
#
# The algorithm is searching for rectangles based on color (HSV), shape and surrounding color (if surrounding color is similar).
#
# Additional improvements can be made by: 
# - checking if squares are close to each other,
# - checking if background color is ~white or
# - even doing something more specific eg. check the position of rectangles.
#
# You can additional colors in `SQUARE_COLORS` dictionary.
#
# Please fix BGR/RGB issue!

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Print dataset
print("List of images:")
print(os.listdir("../input"))

# Load test image
image_path = os.path.join('../input', 'Image1.png')
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Show test image
fig, axs = plt.subplots(1, 1, figsize=(10,10))
axs.imshow(image)

SQUARE_COLORS = {
    'green': ([70,150,0], [80,200,255]),
    'black': ([0,0,0], [255,255,70]),
    'blue': ([10,120,0], [30,180,255]),
    'orange': ([100, 180, 0], [120, 210, 255]),
    'ocean': ([80, 200, 0], [110, 240, 255]),
}

def get_avg_color_of_neighbor(x, y, dist=2):
    neighbor_mean = [image_hsv[y - dist: y + dist, x - dist: x + dist, i].mean() for i in range(3)]
    return np.array(list(map(int, neighbor_mean)))

def check_contour_area(contour, min_area=50):
    return cv2.contourArea(contour) > min_area

def is_rectangle(contour, tolerance=0.03):
    approx = cv2.approxPolyDP(contour, tolerance * cv2.arcLength(contour, True), True)
    return len(approx) == 4

def is_perspective_square(contour):
    _, _, w, h = cv2.boundingRect(contour)
    return h >= w
    
def is_on_wall(contour, max_distance=20):
    x, y, w, h = cv2.boundingRect(contour)
    
    # Get coordinates above & below the rectangle
    y_down = int(y + h + h/2)
    y_up = int(y - h/2)
    x_middle = int(x + w / 2)
    
    # Find average color
    color_up = get_avg_color_of_neighbor(x_middle, y_up)
    color_down = get_avg_color_of_neighbor(x_middle, y_down)
    
    # Check Euclidean distance between up and down colors
    dist = np.linalg.norm(color_up - color_down)
    return dist < max_distance
    # TODO: Check the actual color

def filter_contours(mask):
    filtered_contours = []
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        if check_contour_area(contour) and \
            is_rectangle(contour) and \
            is_perspective_square(contour) and \
            is_on_wall(contour):
            filtered_contours.append(contour)
    return filtered_contours

    
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_with_squares = image.copy()

square_masks = {}
global_square_mask = 0
square_x_color = {}
for color, value in SQUARE_COLORS.items():
    mask = cv2.inRange(image_hsv, np.array(value[0]), np.array(value[1]))
    square_masks[color] = mask
    global_square_mask = cv2.bitwise_or(global_square_mask, mask)
    
    # Get contours
    contours = filter_contours(mask)
    
    # Display contours
    for contour in contours:
        square_x_color[color] = cv2.boundingRect(contour)[0]
        cv2.drawContours(image_with_squares, [contour], 0, 255, -1)

# Find order
import operator
square_x_color = sorted(square_x_color.items(), key=operator.itemgetter(1))
print('Color sequence:')
print(square_x_color)
        
fig, axs = plt.subplots(2, 3, figsize=(20,15))
axs[0][0].imshow(image_with_squares)
axs[0][1].imshow(image_hsv)
axs[0][2].imshow(global_square_mask)
axs[1][0].imshow(square_masks['green'])
axs[1][1].imshow(square_masks['blue'])
axs[1][2].imshow(square_masks['black'])
