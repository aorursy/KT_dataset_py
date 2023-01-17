# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

#RGB to gray conversion 
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

#East China March week 4 
East_china_March_week4 = img.imread('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/East_China-S20200322-E20200331.tif')
gray=rgb2gray(East_china_March_week4)

#East China April week 1 
East_china_April_week1 = img.imread('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/East_China-S20200401-E20200410.tif')
gray1=rgb2gray(East_china_April_week1)


#East China May week 1 
East_china_May_week1 = img.imread('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/East_China-S20200501-E20200510.tif')
gray2=rgb2gray(East_china_May_week1)



# Read East china March Week_4
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.imshow(East_china_March_week4)
ax2 = fig.add_subplot(2,3,2)
ax2.imshow(gray)
ax3 = fig.add_subplot(2,3,3)
ax3.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


# Read East china March Week_4
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.imshow(East_china_April_week1)
ax2 = fig.add_subplot(2,3,2)
ax2.imshow(gray1)
ax3 = fig.add_subplot(2,3,3)
ax3.imshow(gray1, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


# Read East china March Week_4
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.imshow(East_china_May_week1)
ax2 = fig.add_subplot(2,3,2)
ax2.imshow(gray2)
ax3 = fig.add_subplot(2,3,3)
ax3.imshow(gray2, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from skimage import color
from skimage import io


# Read Numpy Files
east_china_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/East_China.npy')
france_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/France.npy')
italy_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/Italy.npy')
usa_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/USA.npy')
west_china_npy=np.load('../input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/npy/West_China.npy')

# Each country have different image size, hence store image size as a variable
east_china_image_size=east_china_npy.shape
france_image_size=france_npy.shape
italy_image_size=italy_npy.shape
usa_image_size=usa_npy.shape
west_china_image_size=west_china_npy.shape

# 1. Raw image to gray conversion
# 2. Load output images into plot figure (5 X 3)

fig = plt.figure(figsize=(50, 50))  # width, height in inches   
for i in range(15):
    sub = fig.add_subplot(5,3, i + 1)
    #sub.imshow(east_china_npy[i,:,:], interpolation='nearest')
    #sub.imshow(gray[i,:,:], interpolation='nearest')
    #gray=rgb2gray(east_china_npy[i])
    gray = color.rgb2gray(usa_npy[i])
    sub.imshow(gray,interpolation='nearest')
    

fig = plt.figure(figsize=(50, 50))  # width, height in inches   
for i in range(15):
    sub = fig.add_subplot(5,3, i + 1)
    gray = color.rgb2gray(usa_npy[i])
    sub.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1,interpolation='nearest')