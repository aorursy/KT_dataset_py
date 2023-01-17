# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from array import array

import numpy as np

import graphlab as gl

gl.canvas.set_target('ipynb')
## constants

TRAIN_DIR = "../input/train/"

TEST_DIR = "../input/test/"

TRAIN_SIZE = 22500

TEST_SIZE = 2500

DEV_RATIO = 0.1

IMAGE_HEIGHT = IMAGE_WIDTH = 128

CHANNELS = 3

OUTPUT_SIZE = 2
image_sframe = gl.image_analysis.load_images(TRAIN_DIR)
image_sarray = image_sframe['image']
resized_image_sarry = gl.image_analysis.resize(image_sarray, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
img1 = resized_image_sarry[0]
img1.pixel_data
data = np.zeros((1000, IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS))

for i in range(1000):

    img = resized_image_sarry[i]

    px_data = img1.pixel_data.reshape(1, IMAGE_HEIGHT*IMAGE_WIDTH*CHANNELS)

    data[i] = px_data