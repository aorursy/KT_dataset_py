# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import cv2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
def get_ex(): return open_image('../input/camseq-semantic-segmentation/0016E5_07959.png')
def plots_f(rows, cols, width, height, **kwargs):

    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(

        rows,cols,figsize=(width,height))[1].flatten())]
plots_f(2, 4, 12, 6, size=224)
# contrast

fig, axs = plt.subplots(1,5,figsize=(12,4))

for scale, ax in zip(np.exp(np.linspace(log(0.5),log(2),5)), axs):

    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')
# brightness

fig, axs = plt.subplots(1,5,figsize=(14,8))

for change, ax in zip(np.linspace(0.1,0.9,5), axs):

    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')
# dihedral

fig, axs = plt.subplots(2,2,figsize=(12,8))

for k, ax in enumerate(axs.flatten()):

    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')

plt.tight_layout()
# tilt

fig, axs = plt.subplots(2,4,figsize=(12,8))

for i in range(4):

    get_ex().tilt(i, 0.4).show(ax=axs[0,i], title=f'direction={i}, fwd')

    get_ex().tilt(i, -0.4).show(ax=axs[1,i], title=f'direction={i}, bwd')
image = cv2.imread('/kaggle/input/camseq-semantic-segmentation/0016E5_08151.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



plt.subplot(1, 2, 1)

plt.title("Original")

plt.imshow(image)



# Cordinates of the 4 corners of the original image

points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])



# Cordinates of the 4 corners of the desired output

# We use a ratio of an A4 Paper 1 : 1.41

points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])

 

# Use the two sets of four points to compute 

# the Perspective Transformation matrix, M    

M = cv2.getPerspectiveTransform(points_A, points_B)





warped = cv2.warpPerspective(image, M, (420,594))



plt.subplot(1, 2, 2)

plt.title("warpPerspective")

plt.imshow(warped)
# Load our new image

image = cv2.imread('/kaggle/input/camseq-semantic-segmentation/0016E5_08029.png', 0)



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.title("Original")

plt.imshow(image)



# Values below 127 goes to 0 (black, everything above goes to 255 (white)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



plt.subplot(3, 2, 2)

plt.title("Threshold Binary")

plt.imshow(thresh1)



# It's good practice to blur images as it removes noise

image = cv2.GaussianBlur(image, (3, 3), 0)



# Using adaptiveThreshold

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.title("Otsu's Thresholding")

plt.imshow(th2)





plt.subplot(3, 2, 5)

# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3)

plt.show()