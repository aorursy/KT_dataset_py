# imports

import os

import ast

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as pc



from random import choice
# set path

train_images_path = "/kaggle/input/global-wheat-detection/train/"

train_metadata_path = "/kaggle/input/global-wheat-detection/train.csv"



# read metadata

train = pd.read_csv(train_metadata_path)
## plot image with bboxes and print metadata



# get random image and its metadata

image_id = choice(train.image_id)

image_metadata = train[train.image_id == image_id]

image = plt.imread("{}.jpg".format(train_images_path + image_id))



# plot image with bboxes

plt.imshow(image)

for xmin, ymin, width, height in image_metadata.bbox.apply(lambda x: ast.literal_eval(x)).values:

    rectangle = pc.Rectangle((xmin,ymin), width, height, fc='none',ec="red")

    plt.gca().add_patch(rectangle)

plt.show()



print("> ID of image:", image_id)



print("> Image width is {} and height is {}".format(image_metadata.width.values[0], image_metadata.height.values[0]))



print("> Location of the recorded image (aka source):", image_metadata.source.values[0])



# num objects

print("> Number of wheat heads in the image:", image_metadata.shape[0])



# average area of bboxes for this image

print("> Average area of bounding boxes: {:.2f}".format(image_metadata.bbox.apply(lambda x: np.prod(ast.literal_eval(x)[2:])).mean()))