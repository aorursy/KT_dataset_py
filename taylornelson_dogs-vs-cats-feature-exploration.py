#load a bunch of useful libraries

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image, ImageFilter

import skimage.feature as sk

import cv2 #OpenCV is a powerful library



#make sure to display images inline

%matplotlib inline
#for giggles, load an image from the input files



#read the file

img = mpimg.imread('../input/train/cat.2.jpg')



#display the image

imgplot = plt.imshow(img)
#the image is now a numpy array with RGB values for each pixel

type(img[0]),img[0][0]
#seems like converting RGB to grayscale and also finding edges & corners are good steps

#going to reload the image with OpenCV



#pull the same image with OpenCV as grayscale (parameter 2 makes it gray)

image = cv2.imread('../input/train/cat.2.jpg')

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_image
corners = cv2.goodFeaturesToTrack(gray_image,maxCorners=20,qualityLevel=15,minDistance=2)
cv2.SIFT()
dir(cv2)
#find Harris corners

dst = cv2.cornerHarris(gray_image,5,25,0.4)

dst
plt.imshow(dst)
#attempting grey-level co-occurence matrix for texture analysis

#techniques include GLCM, LBP, LDP for features related to texture

#glcm = sk.greycomatrix(image,[1],[0,135,90])

#glcm[0]

#other things I will look at are thresholding and also dividing the images into smaller pieces for

#a bag of features

#perhaps I need to flatten and blur