# This code snip fills holes inside large (blue) rocks



import cv2

import  numpy

import matplotlib.pyplot as plt



# Loads a segmented image and isolates larger (blue) rocks

ground_truth = numpy.array(cv2.imread("../input/images/ground/ground0001.png"))

data_blue = ground_truth[:,:,0]



# Creates a 10x10 kernel so as to close any hole smaller than 10x10 pixels inside blue rocks

kernel = numpy.ones((10,10),numpy.uint8)

closing = cv2.morphologyEx(data_blue, cv2.MORPH_CLOSE, kernel)



fig, ax = plt.subplots(1, 2, figsize=(16,9))

ax[0].axis('off')

ax[0].imshow(data_blue)

ax[0].set_title('FIG 1.1 - Original blue component')

ax[1].axis('off')

ax[1].imshow(closing)

ax[1].set_title('FIG 1.2 - Filling holes inside larger rocks')
from PIL import Image

import matplotlib.pyplot as plt

import numpy



# Loads the segmented image and converts it to an array

ground_truth = Image.open("../input/images/ground/ground0001.png")

data = numpy.array(ground_truth)



# Defines different thresholds deciding which large (blue) rocks to keep

data_blue = data[:,:,2]

min_blue_value_1 = 0

min_blue_value_2 = 50

min_blue_value_3 = 100

min_blue_value_4 = 150



blue_1 = (data_blue > min_blue_value_1)

blue_2 = (data_blue > min_blue_value_2)

blue_3 = (data_blue > min_blue_value_3)

blue_4 = (data_blue > min_blue_value_4)



fig, ax = plt.subplots(1, 5, figsize=(16,9))

ax[0].axis('off')

ax[0].imshow(data_blue)

ax[0].set_title('FIG 2.1 - Original blue component')

ax[1].axis('off')

ax[1].imshow(blue_1)

ax[1].set_title('FIG 2.2 - Threshold = 0')

ax[2].axis('off')

ax[2].imshow(blue_2)

ax[2].set_title('FIG 2.3 -Threshold = 50')

ax[3].axis('off')

ax[3].imshow(blue_3)

ax[3].set_title('FIG 2.4 - Threshold = 100')

ax[4].axis('off')

ax[4].imshow(blue_4)

ax[4].set_title('FIG 2.5 - Threshold = 150')



# This code snip only keeps the most relevant green rocks



import cv2

import  numpy

import matplotlib.pyplot as plt

ground_truth = numpy.array(cv2.imread("../input/images/ground/ground0015.png"))

data = numpy.array(ground_truth)



data_green = data[:,:,1]

min_value = 100

h,w = data_green.shape

for y in range(0, h):

    for x in range(0, w):

        data_green[y, x] = 255 if data_green[y, x] >= min_value else 0

        

kernel = numpy.ones((15,15),numpy.uint8)

kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))

opening = cv2.morphologyEx(data_green, cv2.MORPH_OPEN, kernel_circle)

fig, ax = plt.subplots(1, 2, figsize=(16,9))

ax[0].axis('off')

ax[0].imshow(ground_truth)

ax[0].set_title('Original')

ax[1].axis('off')

ax[1].imshow(opening)

ax[1].set_title('Green component without the smallest rocks')
from PIL import Image

import csv

import matplotlib.patches as patches

import matplotlib.pyplot as plt

import numpy



# Loads the bounding boxes file and extracts those from the first frame

bounding_boxes_list = []

with open("../input/bounding_boxes.csv") as bounding_boxes_csv:

    reader = csv.reader(bounding_boxes_csv, delimiter=',')

    next(bounding_boxes_csv) # Skip the header

    for row in reader:

        if row[0] == '1':

            bounding_boxes_list.append(row[1:5])

        else:

            break



# Loads the first segmented image and displays its bounding boxes

ground_truth = numpy.array(Image.open("../input/images/ground/ground0001.png"))

fig,ax = plt.subplots(1)

ax.axis('off')

ax.imshow(ground_truth)

for  bounding_box in bounding_boxes_list:

    bounding_box = list(map(float, bounding_box)) 

    rect = patches.Rectangle((bounding_box[0]-0.5,bounding_box[1]-0.5),bounding_box[2],bounding_box[3],linewidth=2,edgecolor='w',facecolor='none')

    ax.add_patch(rect)

plt.show()