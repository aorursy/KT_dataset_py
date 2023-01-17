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
import numpy as np

import cv2

import matplotlib.pyplot as plt

import os
path="../input/edgedetection/"
name=os.listdir(path)

image_name=path+name[3]
image = cv2.cvtColor(cv2.resize(cv2.imread(image_name),(224,224)), cv2.COLOR_BGR2RGB)
plt.axis("off")

plt.imshow(image)

plt.show()
img = np.array(image)
img.shape
bw_img =  cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.axis("off")

plt.imshow(bw_img,cmap="gray")

plt.show()
h, w = bw_img.shape
# def createFilterX(n):

#     Filter = np.zeros((n, n))

#     for i in range(1,n):

#         for j in range(1,n):

#             grad_x = i / (i*i + j*j)

#             Filter[i,j]=grad

#     return Filter
# def createFilterY(n):

#     Filter = np.zeros((n, n))

#     for i in range(n):

#         for j in range(n):

#             grad_y = j / (i*i + j*j)

#             Filter[i,j]=grad

#     return Filter
# GX = createFilterX(5)

# print(GX)
# define filters

GX = np.array([[1, 0, -1], 

                [2, 0, -2], 

                [1, 0, -1]])  

GY = np.array([[1, 2, 1],

                [0, 0, 0],

                [-1, -2, -1]])  
# define images with 0s

newGX = np.zeros((h, w))

newGY = np.zeros((h, w))

newG = np.zeros((h, w))
# offset by 1(the filter is a 3×3 matrix, the pixels in the first and last rows as well as the first and last columns cannot be estimated so the output image will be a 1 pixel-depth smaller than the original image.)

for i in range(1, h - 1):

    for j in range(1, w - 1):

        GXGrad = np.sum(np.multiply(bw_img[i-1:i+2, j-1:j+2], GX))

        newGX[i - 1, j - 1] = abs(GXGrad)

        GYGrad = np.sum(np.multiply(bw_img[i-1:i+2, j-1:j+2], GY))

        newGY[i - 1, j - 1] = abs(GYGrad)

        # Gradient Edge

        grad=((GXGrad**2) + (GYGrad**2))**0.5  

        newG[i-1,j-1] = grad
plt.axis("off") 

plt.imshow(newG, cmap='gray')

plt.show()
newGn = newG/np.max(newG)
plt.imshow(newGn, cmap='gray')
_,mask_img = cv2.threshold(newGn, 0.35, 1, cv2.THRESH_BINARY_INV)
plt.imshow(mask_img, cmap="gray")