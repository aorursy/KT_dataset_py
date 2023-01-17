# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from PIL import Image 

import pathlib

import imageio

import matplotlib.pyplot as plt

import cv2

from skimage.color import rgb2gray

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

im=Image.open('../input/datasetss/img_000572018.jpg','r')

#im=Image.open('../input/dddddd/img_000082017.jpg','r')

#im=Image.open('../input/datasetsss/img_004772017.jpg','r')

print(im.size)

#im=im.resize((128,128),Image.ANTIALIAS)

print(im.size)

pix=list(im.getdata())

#pixval=[x for sets in pix for x in sets]

#print(pixval)

#im = color.rgb2gray(im)

#print('New image shape: {}'.format(im_gray.shape))

img = np.uint8(im)

edges = cv2.Canny(img,180,250,apertureSize=3)

imgplot = plt.imshow(edges ,cmap = 'gray')

plt.show()

# Any results you write to the current directory are saved as output.