# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# libraries



import numpy as np



import skimage #scikit-image

from skimage import filters, io, morphology, exposure



import skimage.data as data

import skimage.segmentation as seg

import skimage.filters as filters

import skimage.draw as draw

import skimage.color as color



# Plotting modules and settings.

import matplotlib.pyplot as plt

import seaborn as sns





%matplotlib inline

%config InlineBackend.figure_formats = {'png', 'retina'}
# lets concoct an image

# array of zeros

chess_board = np.zeros((9,9))



# change:

# every second element, starting 1st

# to ones



chess_board[::2, 1::2] = 1



# change:

# every second element, starting 1st

# to ones



chess_board[1::2, ::2] = 1

chess_board
plt.imshow(chess_board)

plt.show();
plt.imshow(chess_board, cmap='gray')

plt.show();
# use a built-in image

camera = data.camera()



plt.imshow(camera, cmap='gray')

plt.show();
type(camera)
print("data type: ", camera.dtype, "\n"

     "data shape: ", camera.shape)
# input / output:

from skimage import io

import os

mandrill = io.imread("../input/mandrill.png")

plt.imshow(mandrill)

plt.show();
# file from a URL

# logo = io.imread('http://scikit-image.org/_static/img/logo.png')

# plt.imshow(logo)

# plt.show()
# a digital image is just numeric data! It is a set of numbers with spatial positions

camera
camera.shape
mandrill.shape
# Create an image (5 x 2 pixels) : height 5, width 4, rgb

rgb_image = np.zeros(shape=(5,2,3), dtype=np.uint8) # <- unsigned 8 bit int



# setting the RGB channels

rgb_image[:,:,0] = 255 # Set red value for all pixels

rgb_image[:,:,1] = 0   # Set green value for all pixels

rgb_image[:,:,2] = 0   # Set blue value for all pixels



plt.imshow(rgb_image)

plt.title("A simple RGB image")

plt.show();
# setting the RGB channels

rgb_image[:,:,0] = 255 # Set red value for all pixels

rgb_image[:,:,1] = 255   # Set green value for all pixels

rgb_image[:,:,2] = 0   # Set blue value for all pixels



plt.imshow(rgb_image)

plt.title("Red plus Green equals...")

plt.show();