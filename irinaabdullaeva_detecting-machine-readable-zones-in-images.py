import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import math

# from imutils import paths



import warnings

warnings.filterwarnings('ignore')

#%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')



print(os.listdir("../input"))

sns.set(rc={'figure.figsize' : (22, 20)})

sns.set_style("darkgrid", {'axes.grid' : True})
def showImg(img, cmap=None):

    plt.imshow(img, cmap=cmap, interpolation = 'bicubic')

    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    plt.show()
def prepareImg(img, height):

    "convert given image to grayscale image (if needed) and resize to desired height"

    assert img.ndim in (2, 3)

    if img.ndim == 3:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = img.shape[0]

    factor = height / h

    return cv2.resize(img, dsize=None, fx=factor, fy=factor)
img = prepareImg(cv2.imread('../input/text-to-lines-segmentation/txt2.png', cv2.IMREAD_GRAYSCALE), 600)

print(img.ndim)

print(img.shape)



showImg(img, cmap='gray')
# Blurs an image using a Gaussian filter. The function convolves the source image with the specified Gaussian kernel.

# Gaussian blurring is highly effective in removing gaussian noise from the image.

# We should specify the width and height of kernel which should be positive and odd. 

# We also should specify the standard deviation in X and Y direction, sigmaX and sigmaY respectively. 

# If only sigmaX is specified, sigmaY is taken as same as sigmaX.

blur = cv2.GaussianBlur(img,(3,3),0)

showImg(blur, cmap='gray')
# initialize a rectangular and square structuring kernel

rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))

rectKernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13))

sq1Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

sq2Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))



# The function morphologyEx can perform advanced morphological transformations using an erosion and dilation as basic operations.

# Black Hat (MORPH_BLACKHAT) - It is the difference between the closing of the input image and input image.

blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)

showImg(blackhat, cmap='gray')
# The part of the blackhat is Closing (MORPH_CLOSE) - Closing is reverse of Opening, Dilation followed by Erosion. 

# It is useful in closing small holes inside the foreground objects, or small black points on the object.

close_img = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, rectKernel)

showImg(close_img, cmap='gray')
# Or try Top Hat - It is the difference between input image and Opening of the image.

tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, rectKernel)

showImg(tophat, cmap='gray')
# The part of the dlackhat is Opening (MORPH_OPEN) - Opening is just another name of erosion followed by dilation. 

# It is useful in removing noise

open_img = cv2.morphologyEx(blur, cv2.MORPH_OPEN, sq2Kernel)

showImg(open_img, cmap='gray')
# The next step in detection is to compute the gradient magnitude representation of the blackhat image using the Scharr operator

# Сompute the Scharr gradient along the x-axis of the blackhat image, 

# revealing regions of the image that are not only dark against a light background, but also contain vertical changes in the gradient

gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

gradX = np.absolute(gradX)



# Compute the Scharr gradient along the y-axis of the blackhat image

gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

gradY = np.absolute(gradY)



# We then take this gradient image and scale it back into the range [0, 255] using min/max scaling

(minVal, maxVal) = (np.min(gradX), np.max(gradX))

gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")



(minVal, maxVal) = (np.min(gradY), np.max(gradY))

gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")



showImg(gradX, cmap='gray')

showImg(gradY, cmap='gray')

showImg(gradY+gradX, cmap='gray')
# Then, we apply a closing operation using our rectangular kernel. This closing operation is meant to close gaps in between characters.

close_img2 = cv2.morphologyEx(gradY+gradX, cv2.MORPH_CLOSE, rectKernel)

showImg(close_img2, cmap='gray')
# threshold - If pixel value is greater than a threshold value, it is assigned one value, else it is assigned another value 

# img - source image, which should be a grayscale image. 

# Second argument is the threshold value which is used to classify the pixel values. 

# Third argument is the maxVal which represents the value to be given if pixel value is more than the threshold value. 

# Last - different styles of thresholding

# Returns: threshold value computed, destination image

(_, imgThres) = cv2.threshold(close_img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

showImg(imgThres, cmap='gray')
# thresh1 = cv2.morphologyEx(imgThres, cv2.MORPH_CLOSE, sq2Kernel)

# showImg(thresh1, cmap='gray')

thresh2 = cv2.erode(imgThres, None, iterations=2)

showImg(thresh2, cmap='gray')
def grab_contours(cnts):

    # if the length the contours tuple returned by cv2.findContours

    # is '2' then we are using either OpenCV v2.4, v4-beta, or

    # v4-official

    if len(cnts) == 2:

        cnts = cnts[0]



    # if the length of the contours tuple is '3' then we are using

    # either OpenCV v3, v4-pre, or v4-alpha

    elif len(cnts) == 3:

        cnts = cnts[1]



    # otherwise OpenCV has changed their cv2.findContours return

    # signature yet again and I have no idea WTH is going on

    else:

        raise Exception(("Contours tuple must have length 2 or 3, "

            "otherwise OpenCV changed their cv2.findContours return "

            "signature yet again. Refer to OpenCV's documentation "

            "in that case"))



    # return the actual contours array

    return cnts
def display_contours(contours):

    plt.figure(figsize=(40, 40))

    for i, c in enumerate(contours):

        contour = c[1]

        plt.subplot(40, 3, i+1)  # A grid of 8 rows x 8 columns

        plt.axis('off')

        plt.title("Contour #{0}, size: {1}".format(i, c[0]))

        _ = plt.imshow(contour, cmap='gray')

    plt.show()
components = grab_contours(cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))

minArea= 30

# append components to result

res = []

for c in components:

    # skip small word candidates

    if cv2.contourArea(c) < minArea:

        continue

    # append bounding box and image of word to result list

    currBox = cv2.boundingRect(c) # returns (x, y, w, h)

    (x, y, w, h) = currBox

    currImg = img[y:y+h, x:x+w]

    res.append((currBox, currImg))

len(res)
showImg(img, cmap='gray')
display_contours(sorted(res, key=lambda entry:entry[0][0]))