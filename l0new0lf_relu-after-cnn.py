import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

import os
img = cv.imread('../input/gradpng/gradpng.png')

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img = (img - img.flatten().mean()) / img.flatten().std() # zero-center
plt.imshow(img, cmap='gray')

plt.colorbar()

plt.show()
def relu(X):

    return np.where(X <=0, 0, X)
nonlin_img = relu(img)
plt.imshow(nonlin_img, cmap='gray')

plt.colorbar()

plt.show()