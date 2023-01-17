# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2
im = cv2.imread("../input/tanker.jpg")
print(im)
print(cv2.imshow)
print(im.shape)
im[:, :, (0, 1)] = 0
from matplotlib import pyplot as plt
from skimage.color import rgb2gray 
plt.imshow(im)
red , yellow = im.copy(),im.copy()
print(red)
print(yellow)
red[:,:,(1,2)]=0
plt.imshow(red)
yellow[:,:,2]=0
plt.imshow(yellow)
print(im.shape)
gray = rgb2gray(im)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(gray.shape)
from skimage.filters import threshold_otsu
thresh = threshold_otsu(gray)
binary = gray > thresh
plt.imshow(binary)
print(binary.shape)
from skimage.filters import gaussian_filter

blur_image = gaussian_filter(gray, sigma = 20)
print(blur_image)
plt.imshow(blur_image)
