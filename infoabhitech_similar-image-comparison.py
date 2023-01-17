# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/data/data"))



# Any results you write to the current directory are saved as output.
import skimage

from skimage import io

from skimage.io import imread

from skimage.io import imshow

from skimage.transform import rotate

import matplotlib.pyplot as plt

from skimage.measure import compare_ssim as ssim

from skimage.transform import resize
# Reading images into array

array1 = imread("../input/data/data/image1.jpg")

array2 = imread("../input/data/data/image2.jpg")

array2cpy = imread("../input/data/data/image2.jpg")
fig1, ax1 = plt.subplots(figsize=(2, 2))

fig2, ax2 = plt.subplots(figsize=(2, 2))

print(ax1.imshow(array1))

print(ax2.imshow(array2))
print(array1.shape)

print(array2.shape)
# Resizing both image to same dimension

array1 = resize(array1, (166, 250))
print(array1.shape)

print(array2.shape)
fig1, ax1 = plt.subplots(figsize=(2, 2))

fig2, ax2 = plt.subplots(figsize=(2, 2))

print(ax1.imshow(array1))

print(ax2.imshow(array2))
array1flt = array1.astype('float')

array2flt = array2.astype('float')

array2cpyflt = array2cpy.astype('float')
skimage.measure.compare_ssim(array1flt, array2flt,multichannel=True)
skimage.measure.compare_ssim(array2flt, array2cpyflt,multichannel=True)