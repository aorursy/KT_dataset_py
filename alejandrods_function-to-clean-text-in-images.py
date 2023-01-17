# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from scipy.misc import imread, imsave
import cv2
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave

# Any results you write to the current directory are saved as output.
path_data = '../input/train.csv'
data = pd.read_csv(path_data)
data.head(10)
# Take example
data_example = data.loc[0].values

img_dir = data_example[0]
label = data_example[1]

# Read image
path_join = os.path.join('../input/train', img_dir)
image = imread(path_join)

# Plot image
imgplot = plt.imshow(image)
plt.title(label)
path_join = os.path.join('../input/train', '2b96cac5a.jpg')
image = imread(path_join)

backtorgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
plt.imshow(backtorgb)
def clean_image(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    low_threshold = 20
    high_threshold = 50
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 3  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    y_1 = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope  = (y2 - y1) / (x2 - x1)
            if slope < 0.06 and slope > -0.06:
                region = gray.shape[0] - (gray.shape[0] * 0.30)
                if y1 > region:
                    y_1.append(y1)    
            else:
                continue
    return y_1

data_images = data['Image'].values
data_labels = data['Id'].values

images_preprocess = []
labels_preprocess = []

##############################################################
# NOTE: DELETE data_images[:10] to preprocess all images. 
##############################################################
print('DELETE -> [:10] // in data_images[:10] to preprocess all images')

i = 0
for items in data_images[:10]:
    path_join = os.path.join('../input/train', items)
    image = cv2.imread(path_join)
    image_original = image

    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Crop the image and check if has white pixels
    bottom_percent = 0.25
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    img = image[bottom:image.shape[0], :]

    n_white_pix = np.sum(img >= 250)

    if n_white_pix >= 90000:
        y1 = clean_image(image)

        # Crop image
        if y1:
            min_y1 = min(y1)
            image_original = image_original[0:min_y1, 0:image_original.shape[1]]
            images_preprocess.append(image_original)
            labels_preprocess.append(data_labels[i])
    else:
        images_preprocess.append(image_original)
        labels_preprocess.append(data_labels[i])

    i += 1
    
images_preprocess = np.array(images_preprocess)
labels_preprocess = np.array(labels_preprocess)
print(images_preprocess.shape)
print(labels_preprocess.shape)
path_join = os.path.join('../input/train', '2b96cac5a.jpg')
# image = imread(path_join)
image = cv2.imread(path_join)
image_original = image
plt.imshow(image)
plt.title('BEFORE PREPROCESS')
plt.show()

if len(image.shape) < 3:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Crop the image and check if has white pixels
bottom_percent = 0.25
bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
img = image[bottom:image.shape[0], :]

n_white_pix = np.sum(img >= 250)

if n_white_pix >= 90000:
    y1 = clean_image(image)

    # Crop image
    if y1:
        min_y1 = min(y1)
        image_original = image_original[0:min_y1, 0:image_original.shape[1]]
else:
    pass

plt.imshow(image_original)
plt.title('AFTER PREPROCESS')
plt.show()
