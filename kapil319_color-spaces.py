# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import cv2

import matplotlib.pyplot as plt

import matplotlib.image as mpimg
image = mpimg.imread('/kaggle/input/gradient-and-color-spaces/test6.jpg')
thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

binary = np.zeros_like(gray)
binary[(gray>thresh[0]) & (gray <= thresh[1])] = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(image)

ax1.set_title('Original Image')

ax2.imshow(binary, cmap='gray')

ax2.set_title('Threshold Image')
R = image[:,:,0]

G = image[:,:,1]

B = image[:,:,2]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(R, cmap='gray')

ax1.set_title('R')

ax2.imshow(G, cmap='gray')

ax2.set_title('G')

ax3.imshow(B, cmap='gray')

ax3.set_title('B')
thres = (200, 255)

binary = np.zeros_like(R)

binary[(R > thres[0]) & (R <= thres[1])] = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(R, cmap='gray')

ax1.set_title('R')

ax2.imshow(binary, cmap='gray')

ax2.set_title('R Binary')
hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

H = hls[:,:,0]

L = hls[:,:,1]

S = hls[:,:,2]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(H, cmap='gray')

ax1.set_title('H')

ax2.imshow(L, cmap='gray')

ax2.set_title('L')

ax3.imshow(S, cmap='gray')

ax3.set_title('S')
thres = (90, 255)

binary = np.zeros_like(S)

binary[(S > thres[0]) & (S <= thres[1])] = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(S, cmap='gray')

ax1.set_title('S Channel')

ax2.imshow(binary, cmap='gray')

ax2.set_title('S Binary')
thres = (15, 100)

binary = np.zeros_like(H)

binary[(H > thres[0])  & (H <= thres[1])] = 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (24, 9))

fig.tight_layout()

ax1.imshow(H, cmap='gray')

ax1.set_title('H Channel')

ax2.imshow(binary, cmap='gray')

ax2.set_title('H Binary')