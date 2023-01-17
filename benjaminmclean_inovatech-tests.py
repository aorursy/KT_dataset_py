# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

import pathlib
import imageio

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
test_image_path = "../input/testimages/2020-10-07.jpg"
test_image = imageio.imread(str(test_image_path))

#We can see the dimensions of our image:
print('Original image shape: {}'.format(test_image.shape))

# Coerce the image into grayscale format (if not already)
from skimage.color import rgb2gray
test_image_gray = rescale_frame(test_image, 25)
print('New image shape: {}'.format(test_image_gray.shape))

#As you can see, we load the image as a 3D array, height * width * colors.
# Now, let's plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(test_image)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(test_image_gray, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')

plt.tight_layout()
plt.show()
#Now, to do some basic canny edge detection.

canny_min = 70
canny_max = 150

blur_kernel = np.ones((5, 5), np.float32) / 12
blur_image = cv2.filter2D(test_image, -1, blur_kernel)

origional_edges = cv2.Canny(np.uint8(test_image), canny_min, canny_max)
blur_edge = cv2.Canny(np.uint8(blur_image), canny_min, canny_max)
                            
#cmap is defining our color map,
#xticks, yticks are for ui x, y axis ticks.

fig = plt.figure(figsize=(20, 20))

plt.subplot(1,2,1)
plt.imshow(origional_edges, cmap="gray")
plt.title('Origional Image as Input')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(blur_edge, cmap="gray")
plt.title('Blured Image as Input')
plt.xticks([])
plt.yticks([])

import math

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 2     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 4 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments
newImage = rescale_frame(blur_edge, 15)
line_image = np.copy(newImage) # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
# Output "lines" is an array containing endpoints of detected line segments
rho = 0.1
theta = np.pi / 720
threshold=0
minLineLength=1000 #does nothing???
maxLineGap=50

print(blur_edge.shape, newImage.shape)

lines = cv2.HoughLines(newImage,1,np.pi/720,50)
if lines is not None:
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        b = [math.cos(theta)*rho, math.sin(theta)*rho]
        m = [-math.sin(theta), math.cos(theta)]
        x1 = int(round(b[0]))
        y1 = int(round(b[1]))
        x2 = int(round(b[0] - m[0]*1000))
        y2 = int(round(b[1] - m[1]*1000))
        
        line_image = cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 1)

        
fig = plt.figure(figsize=(20, 20))

plt.subplot(1,2,1)
plt.imshow(newImage, cmap="gray")
plt.title('Hough Input')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(line_image, cmap="gray")
plt.title('Hough Lines')
plt.xticks([])
plt.yticks([])
