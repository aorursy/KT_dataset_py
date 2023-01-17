import cv2

import numpy as np

import math

import matplotlib.pyplot as plt

%matplotlib inline
IMAGE = '../input/lena.bmp'

IMAGE_GRAY = '../input/lena.gray.bmp'
#OpenCV will import all images (grayscale or color) as having 3 channels

color_img_as_grayscale = cv2.imread(IMAGE, 0) #0 for read as grayscale(single channel), -1 for read as it is, default 1 for read as color

print("Shape of lena.bmp after converting to/read as single channel (grayscale) i.e., color_img_as_grayscale : {}".format(color_img_as_grayscale.shape))
#Without cmap='gray argument, matplotlib will try to plot the gray image as a RGB image. This is wrong way of plotting image

plt.imshow(color_img_as_grayscale)

plt.title("Wrong")
#The correct way of plotting a grayscale image using matplotlib goes like this

plt.imshow(color_img_as_grayscale, cmap='gray')

plt.title("Right")
color_img_bgr = cv2.imread(IMAGE) #default 1 for read as color

print("Shape of lena.bmp as it is (color), i.e., color_img : {}".format(color_img_bgr.shape))
# Example of how matplotlib displays color images from OpenCV incorrectly

plt.imshow(color_img_bgr)

plt.title("Wrong : How OpenCV images (BGR) display in Matplotlib (RGB)")
#images are stored as numpy arrays

type(color_img_bgr)
#Right way of doing it

color_img_rgb = color_img_bgr[:,:,::-1] #slicing only the channel component to reverse its order: BGR reversed is RGB!

plt.imshow(color_img_rgb)

plt.title("Right")
img = cv2.imread(IMAGE_GRAY)

print("Shape of lena.gray.bmp i.e., img : {}".format(img.shape))
plt.imshow(img)

plt.title("lena.gray.bmp")
#Didn't do numpy array slicing above since each pixel has the same value in all 3 channels.

img
def reduce_intensity(img, factor):

    """Quantize by the specified factor"""

    return np.multiply(np.floor(np.divide(img, factor)).astype(int), factor)
reduced_intensity_image = reduce_intensity(color_img_rgb, 128)

plt.imshow(reduced_intensity_image)
from scipy import ndimage

def neighborhood_averaging(img, filter_size):

    """If filter_size is 3, then average of 3x3 is calculated. Centre pixel is not ignored"""

    #return ndimage.generic_filter(input=img, function=np.nanmean, size=filter_size, mode='constant', cval=np.NaN)

    return cv2.blur(img, (filter_size, filter_size))
fig = plt.figure(figsize=(25, 25))



img1 = fig.add_subplot(1,4,1) #1 rows, 4 coulmns, image fills the 1st column

img1.imshow(color_img_rgb)

plt.title("Original image")



img2 = fig.add_subplot(1,4,2) #1 rows, 4 coulmns, image fills the 2nd column

img2.imshow(neighborhood_averaging(color_img_rgb, 3))

plt.title("3x3 neighbourhood averaged image")



img3 = fig.add_subplot(1,4,3) #1 rows, 4 coulmns, image fills the 3rd column

img3.imshow(neighborhood_averaging(color_img_rgb, 10))

plt.title("10x10 neighbourhood averaged image")



img4 = fig.add_subplot(1,4,4) #1 rows, 4 coulmns, image fills the 4th column

img4.imshow(neighborhood_averaging(color_img_rgb, 20))

plt.title("20x20 neighbourhood averaged image")
fig = plt.figure(figsize=(25, 25))



img1 = fig.add_subplot(1,4,1) #1 rows, 4 coulmns, image fills the 1st column

img1.imshow(color_img_as_grayscale, cmap='gray')

plt.title("Original image")



img2 = fig.add_subplot(1,4,2) #1 rows, 4 coulmns, image fills the 2nd column

img2.imshow(neighborhood_averaging(color_img_as_grayscale, 3), cmap='gray')

plt.title("3x3 neighbourhood averaged image")



img3 = fig.add_subplot(1,4,3) #1 rows, 4 coulmns, image fills the 3rd column

img3.imshow(neighborhood_averaging(color_img_as_grayscale, 10), cmap='gray')

plt.title("10x10 neighbourhood averaged image")



img4 = fig.add_subplot(1,4,4) #1 rows, 4 coulmns, image fills the 4th column

img4.imshow(neighborhood_averaging(color_img_as_grayscale, 20), cmap='gray')

plt.title("20x20 neighbourhood averaged image")
def image_rotate(img, angle):

    """Rotates the image by angle degrees"""

    rows, cols, _ = img.shape

    rotation = cv2.getRotationMatrix2D(center=(cols/2,rows/2), angle=angle, scale=1)

    rotated = cv2.warpAffine(src=img, M=rotation, dsize=(cols, rows))

    return rotated
fig = plt.figure(figsize=(20, 20))



img1 = fig.add_subplot(1,3,1)

img1.imshow(color_img_rgb)

plt.title("Original image")



img2 = fig.add_subplot(1,3,2)

img2.imshow(image_rotate(color_img_rgb, 45))

plt.title("45 degree rotated image")



img3 = fig.add_subplot(1,3,3)

img3.imshow(image_rotate(color_img_rgb, 90))

plt.title("90 degree rotated image")
def shrink_resolution(img, factor):

    """Reduces resolution of an image by reducing and re enlarging it using the average of the neighbouring pixels"""

    shrunk = cv2.resize(img, (0,0), None, fx=1.0/factor, fy=1.0/factor, interpolation=cv2.INTER_AREA)

    print("Shape of reduced/shrunk image = {}".format(shrunk.shape))

    re_enlarged= cv2.resize(shrunk, (0,0), None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    print("Shape of re-enlarged image = {}".format(re_enlarged.shape))

    return (shrunk, re_enlarged)
fig = plt.figure(figsize=(20, 20))



#shrinking and re-enlarging by a factor of 10 using 10x10 blocks

(shrunk, re_enlarged) = shrink_resolution(color_img_rgb, 10)



img1 = fig.add_subplot(1,3,1)

img1.imshow(color_img_rgb)

plt.title("Original image")



img2 = fig.add_subplot(1,3,2)

img2.imshow(shrunk)

plt.title("Shrunk image")



img3 = fig.add_subplot(1,3,3)

img3.imshow(re_enlarged)

plt.title("Re-enlarged image")