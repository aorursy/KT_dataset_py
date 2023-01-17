# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install imutils

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.utils import shuffle

import cv2

import imutils

import numpy as np

import matplotlib.pyplot as plt

import time

from os import listdir



%matplotlib inline
def crop_brain_contour(image, plot=False):

    

    #import imutils

    #import cv2

    #from matplotlib import pyplot as plt

    

    # Convert the image to grayscale, and blur it slightly

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)



    # Threshold the image, then perform a series of erosions +

    # dilations to remove any small regions of noise

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)

    thresh = cv2.dilate(thresh, None, iterations=2)



    # Find contours in thresholded image, then grab the largest one

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    



    # Find the extreme points

    extLeft = tuple(c[c[:, :, 0].argmin()][0])

    extRight = tuple(c[c[:, :, 0].argmax()][0])

    extTop = tuple(c[c[:, :, 1].argmin()][0])

    extBot = tuple(c[c[:, :, 1].argmax()][0])

    

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            



    if plot:

        plt.figure()



        plt.subplot(1, 2, 1)

        plt.imshow(image)

        

        plt.tick_params(axis='both', which='both', 

                        top=False, bottom=False, left=False, right=False,

                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        

        plt.title('Original Image')

            

        plt.subplot(1, 2, 2)

        plt.imshow(new_image)



        plt.tick_params(axis='both', which='both', 

                        top=False, bottom=False, left=False, right=False,

                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)



        plt.title('Cropped Image')

        

        plt.show()

    

    return new_image
ex_img = cv2.imread('input/brain-mri-images-for-brain-tumor-detection/yes/Y1.jpg')

#ex_new_img = crop_brain_contour(ex_img, True)

print(ex_img)
!pip install opencv
