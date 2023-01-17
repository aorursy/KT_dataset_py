# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.chdir("../input")
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray
import glob
from skimage.io import imread

from PIL import Image 
example_file=glob.glob(r"../input/wint_sky.gif")[0] #enter path of your image file
im=imread(example_file, as_gray=True)
plt.imshow(im, cmap=plt.get_cmap('gray'))
plt.show()

blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=.1)
blobs_log[:,2]= blobs_log[:,2]*sqrt(2)
numrows=len(blobs_log)
print("Number of stars counted: ", numrows)

fig, ax= plt.subplots(1,1)
plt.imshow(im, cmap=plt.get_cmap('gray'))
for blob in blobs_log:
	y, x, r=blob
	c=plt.Circle((x,y), r+5, color='lime', linewidth='2', fill=False)
	ax.add_patch(c)

plt.show()

