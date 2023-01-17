import cv2

import numpy as np

from pylab import *

import matplotlib.pyplot as plt

from skimage import io
# Read Image 01

image_01 = cv2.imread('../input/puppy-image/puppy_01.jpg')

# Read image 02

image_02 = cv2.imread('../input/puppy-image/puppy_02.jpg')
# Define alpha and Beta

alpha = 0.30

beta = 0.70
# Blend images

blend_images = cv2.addWeighted(image_01, alpha, image_02, beta, 0.0)
# Image show

plt.figure(figsize = (10, 8))

io.imshow(blend_images)
image = cv2.imread('../input/puppy-image/puppy_01.jpg')
# Create a dummy image that stores different contrast and brightness

new_image = np.zeros(image.shape, image.dtype)
# Brightness and contrest parameters

contrast = 3.0

bright = 2
# Change the contrast and brightness

for x in range(image.shape[0]):

    for y in range(image.shape[1]):

        for z in range(image.shape[2]):

            new_image[x, y, z] = np.clip(contrast * image[x, y, z] + bright, 0, 255)
# Show First Image

figure(0)

plt.figure(figsize = (10, 8))

plt.title("Before Change Image Contrast and Brightness")

io.imshow(image)



# Show Second Image

figure(1)

plt.figure(figsize = (10, 8))

plt.title("After Change Image Contrast and Brightness")

io.imshow(new_image)
# Define Font

font = cv2.FONT_HERSHEY_SIMPLEX
# Write on the image

plt.figure(figsize = (10, 8))

cv2.putText(image, "I am a Cute Puppy", (10, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

io.imshow(image)
#Read images for different blurring purposes

image_Original = cv2.imread("../input/puppy-image/puppy_01.jpg")

image_MedianBlur = cv2.imread("../input/puppy-image/puppy_01.jpg")

image_GaussianBlur = cv2.imread("../input/puppy-image/puppy_01.jpg")

image_BilateralBlur = cv2.imread("../input/puppy-image/puppy_01.jpg")



#Blur images

image_MedianBlur=cv2.medianBlur(image_MedianBlur,9)

image_GaussianBlur=cv2.GaussianBlur(image_GaussianBlur,(9,9),10)

image_BilateralBlur=cv2.bilateralFilter(image_BilateralBlur,9,100,75)
#Show images

figure(0)

plt.title("Orginal Image")

io.imshow(image_Original)



figure(1)

plt.title("Median Blur Image")

io.imshow(image_MedianBlur)



figure(2)

plt.title("Gaussian Blur Image")

io.imshow(image_GaussianBlur)



figure(3)

plt.title("Bilateral Blur Image")

io.imshow(image_BilateralBlur)
#Read image

image = cv2.imread("../input/puppy-image/puppy_01.jpg")

#Define erosion size

s1 = 0

s2 = 10

s3 = 10
#Define erosion type

t1 = cv2.MORPH_RECT

t2 = cv2.MORPH_CROSS

t3 = cv2.MORPH_ELLIPSE

#Define and save the erosion template

tmp1 = cv2.getStructuringElement(t1, (2*s1 + 1, 2*s1+1), (s1, s1))

tmp2= cv2.getStructuringElement(t2, (2*s2 + 1, 2*s2+1), (s2, s2))

tmp3 = cv2.getStructuringElement(t3, (2*s3 + 1, 2*s3+1), (s3, s3))

#Apply the erosion template to the image and save in different variables

final1 = cv2.erode(image, tmp1)

final2 = cv2.erode(image, tmp2)

final3 = cv2.erode(image, tmp3)
#Show all the images with different erosions

figure(0)

io.imshow(final1)

figure(1)

io.imshow(final2)

figure(2)

io.imshow(final3)
#Define dilation size

d1 = 0

d2 = 10

d3 = 20

#Define dilation type

t1 = cv2.MORPH_RECT

t2 = cv2.MORPH_CROSS

t3 = cv2.MORPH_ELLIPSE

#Store the dilation templates

tmp1 = cv2.getStructuringElement(t1, (2*d1 + 1, 2*d1+1), (d1, d1))

tmp2 = cv2.getStructuringElement(t2, (2*d2 + 1, 2*d2+1), (d2, d2))

tmp3 = cv2.getStructuringElement(t3, (2*d3 + 1, 2*d3+1), (d3, d3))

#Apply dilation to the images

final1 = cv2.dilate(image, tmp1)

final2 = cv2.dilate(image, tmp2)

final3 = cv2.dilate(image, tmp3)
#Show the images

figure(0)

io.imshow(final1)

figure(1)

io.imshow(final2)

figure(2)

io.imshow(final3)
#Define threshold types

"""

0 - Binary

1 - Binary Inverted

2 - Truncated

3 - Threshold To Zero

4 - Threshold To Zero Inverted

"""
#Apply different thresholds and save in different variables

_, img1 = cv2.threshold(image, 50, 255, 0 )

_, img2 = cv2.threshold(image, 50, 255, 1 )

_, img3 = cv2.threshold(image, 50, 255, 2 )

_, img4 = cv2.threshold(image, 50, 255, 3 )

_, img5 = cv2.threshold(image, 50, 255, 4 )



#Show the different threshold images

figure(0)

io.imshow(img1) #Prints Binary Image

figure(1)

io.imshow(img2) #Prints Binary Inverted Image

figure(2)

io.imshow(img3) #Prints Truncated Image

figure(3)

io.imshow(img4) #Prints Threshold to Zero Image

figure(4)

io.imshow(img5) #Prints Threshold to Zero Inverted Image
#Apply gaussian blur

cv2.GaussianBlur(image, (3, 3), 0)

#Convert image to grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply Sobel method to the grayscale image

grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1,

delta=0, borderType=cv2.BORDER_DEFAULT) #Horizontal Sobel Derivation

grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1,

delta=0, borderType=cv2.BORDER_DEFAULT) #Vertical Sobel Derivation

abs_grad_x = cv2.convertScaleAbs(grad_x)

abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

#Apply both

#Show the image

io.imshow(grad)#View the image
#Convert to grayscale

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply equalize histogram

image_eqlzd = cv2.equalizeHist(image) #Performs Histogram Equalization

#Show both images

figure(0)

io.imshow(image)

figure(1)

io.imshow(image_eqlzd)

figure(2)

io.imshow(image_eqlzd)