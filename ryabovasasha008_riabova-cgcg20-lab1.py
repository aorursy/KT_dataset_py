import numpy as np

import pandas as pd 

import glob,cv2

import random

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
folderlist = glob.glob("../input/*")  

img = cv2.imread(folderlist[0])

plt.imshow(img)

plt.title('Original image')

plt.show()

def read_as_gray(image):

    img_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY ) 

    rgb_image_grayscale = cv2.cvtColor(img_grayscale, cv2.COLOR_BGR2RGB)

    return rgb_image_grayscale
img_gray = read_as_gray(img)

plt.imshow(img_gray)

plt.title('Grayscale image')

plt.show()
def read_as_YCrCb(image):

    img_ycrcb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)

    img_ycrcb_rgb = cv2.cvtColor(img_ycrcb,cv2.COLOR_BGR2RGB)

    return img_ycrcb_rgb
img_ycrcb = read_as_YCrCb(img)

plt.imshow(img_ycrcb)

plt.title('YCrCb image')

plt.show()
def read_as_red(image):

    b = image.copy()

    b[:, :, 1] = 0

    b[:, :, 2] = 0

    return b
img_red = read_as_red(img)

plt.imshow(img_red)

plt.title('Red image')

plt.show()
def read_as_green(image):

    b = image.copy()

    b[:, :, 0] = 0

    b[:, :, 2] = 0

    return b
img_green = read_as_green(img)

plt.imshow(img_green)

plt.title('Green image')

plt.show()
def read_as_blue(image):

    b = image.copy()

    b[:, :, 0] = 0

    b[:, :, 1] = 0

    return b
img_blue = read_as_blue(img)

plt.imshow(img_blue)

plt.title('Blue image')

plt.show()