# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
img_rgb = cv2.imread('/kaggle/input/safranbolu-sokak/safranbolu_sokak.jpg')





class Cartoonizer:

    """Cartoonizer effect

        A class that applies a cartoon effect to an image.

        The class uses a bilateral filter and adaptive thresholding to create

        a cartoon effect.

    """

    def __init__(self):

        pass

    def render(self, input_image):

        #img_rgb = cv2.resize(img_rgb, (600,206))

        numDownSamples = 1       # number of downscaling steps

        numBilateralFilters = 10  # number of bilateral filtering steps

        # -- STEP 1 --

        # downsample image using Gaussian pyramid

        img_color = input_image

        for _ in range(numDownSamples):

            img_color = cv2.pyrDown(img_color)

        img_transform1 = img_color

        # repeatedly apply small bilateral filter instead of applying

        # one large filter

        for _ in range(numBilateralFilters):

            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        img_transform2 = img_color

        # upsample image to original size

        for _ in range(numDownSamples):

            img_color = cv2.pyrUp(img_color)

        img_transform3 = img_color

        # -- STEPS 2 and 3 --

        # convert to grayscale and apply median blur

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        img_transform4 = img_gray

        img_blur = cv2.medianBlur(img_gray, 3)

        img_transform5 = img_blur

        # -- STEP 4 --

        # detect and enhance edges

        img_edge = cv2.adaptiveThreshold(img_blur, 255,

                                         cv2.ADAPTIVE_THRESH_MEAN_C,

                                         cv2.THRESH_BINARY, 5, 2)

        

        # -- STEP 5 --

        # convert back to color so that it can be bit-ANDed with color image

        (x,y,z) = img_color.shape

        img_edge = cv2.resize(img_edge,(y,x))

        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        kernel = np.ones((1,1),np.uint8)

        img_edge = cv2.dilate(img_edge,kernel,iterations = 1)

        img_transform6 = img_edge

        return  cv2.bitwise_and(img_color, img_edge),img_transform1,img_transform2,img_transform3,img_transform4,img_transform5,img_transform6



tmp_canvas = Cartoonizer()

res,img_transform1,img_transform2,img_transform3,img_transform4,img_transform5,img_transform6 = tmp_canvas.render(img_rgb)

fig = plt.figure()

#fig.set_figheight(15)

fig.set_figwidth(23)

a = fig.add_subplot(2, 4, 1)

imgplot = plt.imshow(img_rgb)

plt.axis('off')

a.set_title('Original')

a = fig.add_subplot(2, 4, 2)

imgplot = plt.imshow(img_transform1)

plt.axis('off')

a.set_title('First Step')

a = fig.add_subplot(2, 4, 3)

imgplot = plt.imshow(img_transform2)

plt.axis('off')

a.set_title('Second Step')

a = fig.add_subplot(2, 4, 4)

imgplot = plt.imshow(img_transform3)

plt.axis('off')

a.set_title('Third Step')

a = fig.add_subplot(2, 4, 5)

imgplot = plt.imshow(img_transform4)

plt.axis('off')

a.set_title('Fourth Step')

a = fig.add_subplot(2, 4, 6)

imgplot = plt.imshow(img_transform5)

plt.axis('off')

a.set_title('Fifth Step')

a = fig.add_subplot(2, 4, 7)

imgplot = plt.imshow(img_transform6)

plt.axis('off')

a.set_title('Sixth Step')

a = fig.add_subplot(2, 4, 8)

imgplot = plt.imshow(res)

plt.axis('off')

a.set_title('Cartonized')



fig.set_figwidth(50)

a = fig.add_subplot(1, 1, 1)

imgplot=plt.imshow(res)