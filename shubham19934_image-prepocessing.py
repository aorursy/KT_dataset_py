import numpy as np

from skimage.io import imread

from skimage.color import rgb2gray

import matplotlib.pyplot as plt

#from skimage.filters import threshold_adaptive

from cv2 import adaptiveThreshold

import cv2



sample_files = ['../input/denoising-dirty-documents/train/101.png', '../input/denoising-dirty-documents/train/11.png', '../input/denoising-dirty-documents/train/120.png']



def denoiseimage(inp_path):

    img = rgb2gray(imread(inp_path))

    block_size = 255

    #apply adaptive threshold.

    binary_adaptive = adaptiveThreshold(img, block_size, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    return binary_adaptive



fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(25,40))

for index, file in enumerate(sample_files):

    noise_reduced_file = denoiseimage(file)

    ax[index][0].imshow(imread(file), cmap="gray")

    ax[index][1].imshow(noise_reduced_file, cmap="gray")

    

plt.tight_layout()

plt.show()
from skimage.morphology import binary_closing



fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(25,40))

kernel=[[1,1],[1,1]]

for index, file in enumerate(sample_files):

    noise_reduced_file = binary_closing(denoiseimage(file), kernel)

    ax[index][0].imshow(imread(file), cmap="gray")

    ax[index][1].imshow(noise_reduced_file, cmap="gray")

    

plt.tight_layout()

plt.show()
inp_path = '../input/denoising-dirty-documents/train/11.png'
img = cv2.imread(inp_path)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)



fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(25,40))



ax[0].imshow(img, cmap="gray")

ax[1].imshow(thresh, cmap="gray")



plt.tight_layout()

plt.show()
BLUE = [255,0,0]



img1 = cv2.imread(inp_path)

img1 = denoiseimage(inp_path)

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)

reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)

reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)

wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)

constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)



fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(25,30))

ax[0][0].imshow(img1, cmap="gray")

ax[0][1].imshow(replicate, cmap="gray")

ax[1][0].imshow(reflect, cmap="gray")

ax[1][1].imshow(reflect101, cmap="gray")

ax[2][0].imshow(wrap, cmap="gray")

ax[2][1].imshow(constant, cmap="gray")



# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')

# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')

# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')

# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')

# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')

# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')



plt.show()
np.zeros((3,2))
img = cv2.imread(inp_path)

normalizedImg = np.zeros((200, 400))

normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)

#cv2.imshow('dst_rt', normalizedImg)



fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(25,40))

ax[0].imshow(img, cmap="gray")

ax[1].imshow(normalizedImg, cmap="gray")

plt.tight_layout()

plt.show()
img = cv2.imread(inp_path)

img = cv2.cvtColor(normalizedImg, cv2.COLOR_BGR2GRAY) 



thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,5)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(25,40))



ax[0].imshow(img, cmap="gray")

ax[1].imshow(thresh, cmap="gray")

plt.tight_layout()

plt.show()
img = cv2.imread(inp_path,0)

kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)

dilation = cv2.dilate(img,kernel,iterations = 1)
import cv2

img = cv2.imread(inp_path)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)