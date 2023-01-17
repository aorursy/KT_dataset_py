#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR4sC37dGKY4dEZSVixMP53f2oc2X5wP2QLi8lRcmuxPuo_7SqD&usqp=CAU',width=400,height=400)
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
def get_ex(): return open_image('../input/bacteria-detection-with-darkfield-microscopy/images/011.png')
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
!ls /kaggle/input/bacteria-detection-with-darkfield-microscopy
INPUT = "/kaggle/input/bacteria-detection-with-darkfield-microscopy"
import cv2

import numpy as np

import matplotlib.pyplot as plt



def visualize(filename):

    image_path = f"{INPUT}/images/{filename}.png"

    mask_path = f"{INPUT}/masks/{filename}.png"

    image = cv2.imread(image_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

    mask_applied = image.copy()

    mask_applied[mask == 1] = [255, 0, 0]

    mask_applied[mask == 2] = [255, 255, 0]

    

    out = image.copy()

    mask_applied = cv2.addWeighted(mask_applied, 0.5, out, 0.5, 0, out)

    

    fig = plt.figure(figsize=(20, 20))

    ax1 = fig.add_subplot(1,3,1)

    ax1.imshow(image)

    ax2 = fig.add_subplot(1,3,2)

    ax2.imshow(mask, cmap="gray")

    ax3 = fig.add_subplot(1,3,3)

    ax3.imshow(mask_applied)
visualize("011")
# Load our new image

image = cv2.imread('/kaggle/input/bacteria-detection-with-darkfield-microscopy/images/011.png', 0)



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
image = cv2.imread('/kaggle/input/bacteria-detection-with-darkfield-microscopy/images/011.png')

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
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'data:image/jpeg;base64,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',width=400,height=400)