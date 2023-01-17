import re

import math

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image
cols, rows = 4, 3

def grid_display(list_of_images, no_of_columns=2, figsize=(15,15), title = False):

    fig = plt.figure(figsize=figsize)

    column = 0

    z = 0

    for i in range(len(list_of_images)):

        column += 1

        #  check for end of column and create a new figure

        if column == no_of_columns+1:

            fig = plt.figure(figsize=figsize)

            column = 1

        fig.add_subplot(1, no_of_columns, column)

        if title:

            if i >= no_of_columns:

                plt.title(titles[z])

                z +=1

            else:

                plt.title(titles[i])

        plt.imshow(list_of_images[i])

        plt.axis('off')
TRAIN = '../input/malaria-parasite-image-malaria-species/Falciparum/img/'

IMAGE_SIZE = 1024

n_imgs = 12

img_filenames = os.listdir(TRAIN)[:n_imgs]

img_filenames[:3]

image=[]

for file_name in img_filenames:

    img = cv2.imread(TRAIN +file_name)

    image.append(img)

grid_display(image, 4, (15,15))
TRAIN = '../input/malaria-parasite-image-malaria-species/Malariae/img/'

IMAGE_SIZE = 1024

n_imgs = 12

img_filenames = os.listdir(TRAIN)[:n_imgs]

img_filenames[:3]

image=[]

for file_name in img_filenames:

    img = cv2.imread(TRAIN +file_name)

    image.append(img)

grid_display(image, 4, (15,15))
TRAIN = '../input/malaria-parasite-image-malaria-species/Ovale/img/'

IMAGE_SIZE = 1024

n_imgs = 12

img_filenames = os.listdir(TRAIN)[:n_imgs]

img_filenames[:3]

image=[]

for file_name in img_filenames:

    img = cv2.imread(TRAIN +file_name)

    image.append(img)

grid_display(image, 4, (15,15))
TRAIN = '../input/malaria-parasite-image-malaria-species/Vivax/img/'

IMAGE_SIZE = 1024

n_imgs = 12

img_filenames = os.listdir(TRAIN)[:n_imgs]

img_filenames[:3]

image=[]

for file_name in img_filenames:

    img = cv2.imread(TRAIN +file_name)

    image.append(img)

grid_display(image, 4, (15,15))
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    #Reducing Noise

    result = cv2.fastNlMeansDenoisingColored(img,None,20,10,7,21)

    image_all.append(result)

grid_display(image_all, 4, (15,15), title = True)
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

#     #Gaussian Blur

    blur_image = cv2.GaussianBlur(img, (7,7), 0)

    image_all.append(blur_image)

grid_display(image_all, 4, (15,15), title = True)
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    #Adjusted contrast

    contrast_img = cv2.addWeighted(img, 2.005, np.zeros(img.shape, img.dtype), 0, 0)

    image_all.append(contrast_img)

grid_display(image_all, 4, (15,15), title = True)
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

'../input/malaria-parasite-image-malaria-species/Falciparum/img/1603223711-0003-T_R.tif']

image_all=[]

titles = ['original Falciparum', 'Adaptive thresholding', "Binary thresholding"]

for image_id in image_list:

    img = cv2.imread(image_id)

    image_all.append(img)

    #Adaptive Thresholding..

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    image_all.append(thresh1)

    #Binary Thresholding...

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    res, thresh = cv2.threshold(hsv[:, :, 0], 20, 255, cv2.THRESH_BINARY_INV)

    image_all.append(thresh)

grid_display(image_all, 3, (15,15), title = True)
image_list = ['../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0036-S_R.tif',

'../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['original Vivax Image', 'Adaptive thresholding', "Binary thresholding"]

for image_id in image_list:

    img = cv2.imread(image_id)

    image_all.append(img)

    #Adaptive Thresholding..

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

    image_all.append(thresh1)

    #Binary Thresholding...

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    res, thresh = cv2.threshold(hsv[:, :, 0], 125, 255, cv2.THRESH_BINARY_INV)

    image_all.append(thresh)

grid_display(image_all, 3, (15,15), title = True)
img = cv2.imread('../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif', 0)

# global thresholding

ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)



# Otsu's thresholding

ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# Otsu's thresholding after Gaussian filtering

blur = cv2.GaussianBlur(img,(5,5),0)

ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



# plot all the images and their histograms

images = [img, 0, th1,

          img, 0, th2,

          blur, 0, th3]

titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',

          'Original Noisy Image','Histogram',"Otsu's Thresholding",

          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

plt.figure(figsize=(15,10))

for i in range(3):

    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')

    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)

    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')

    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()
image_all = []

img = cv2.imread('../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0036-S_R.tif', 1)

image_all.append(img)

# Initiate ORB detector

orb = cv2.ORB_create()

# find the keypoints with ORB

kp = orb.detect(img,None)

# compute the descriptors with ORB

kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation

img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

# plt.imshow(img2)

# plt.show()

img2 = img2[900:1400,1200:1600]

image_all.append(img2)

grid_display(image_all, 3, (15,15), title = False)
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    img_neg = 1 - img

    image_all.append(img_neg)

grid_display(image_all, 4, (25,25), title = True)
import albumentations as alb

chosen_image = cv2.imread('../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0036-S_R.tif')

albumentation_list = [alb.RandomSunFlare(p=1), alb.RandomFog(p=1), alb.RandomBrightness(p=1),

                      alb.RandomCrop(p=1,height = 512, width = 512), alb.Rotate(p=1, limit=90),

                      alb.RGBShift(p=1), alb.RandomSnow(p=1),

                      alb.HorizontalFlip(p=1), alb.VerticalFlip(p=1), alb.RandomContrast(limit = 0.5,p = 1),

                      alb.HueSaturationValue(p=1,hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]



img_matrix_list = []

bboxes_list = []

for aug_type in albumentation_list:

    img = aug_type(image = chosen_image)['image']

    img_matrix_list.append(img)



img_matrix_list.insert(0,chosen_image)    



titles_list = ["Original","RandomSunFlare","RandomFog","RandomBrightness","RandomCrop","Rotate", "RGBShift", "RandomSnow","HorizontalFlip", "VerticalFlip", "RandomContrast","HSV"]



##reminder of helper function

def plot_multiple_img(img_matrix_list, title_list, ncols, main_title=""):

    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=3, ncols=ncols, squeeze=False)

    fig.suptitle(main_title, fontsize = 30)

    fig.subplots_adjust(wspace=0.3)

    fig.subplots_adjust(hspace=0.3)

    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):

        myaxes[i // ncols][i % ncols].imshow(img)

        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)

        plt.axis('off')

    plt.show()

    

plot_multiple_img(img_matrix_list, titles_list, ncols = 4,main_title="Different Types of Augmentations")
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(img,kernel,iterations = 1)

    image_all.append(erosion)

grid_display(image_all, 4, (15,15), title = True)
image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    kernel = np.ones((5,5),np.uint8)

    dilation = cv2.dilate(img,kernel,iterations = 1)

    image_all.append(dilation)

grid_display(image_all, 4, (15,15), title = True)

image_list = ['../input/malaria-parasite-image-malaria-species/Falciparum/img/1701151546-0013-R_T.tif',

              '../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',

              '../input/malaria-parasite-image-malaria-species/Ovale/img/1707180816-0019-S.tif',

             '../input/malaria-parasite-image-malaria-species/Vivax/img/1709041080-0014-R.tif']

image_all=[]

titles = ['Falciparum', 'Malariae', "Ovale", 'Vivax']

for image_id in image_list:

    img = cv2.imread(image_id)

    kernel = np.ones((5,5),np.uint8)

    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    image_all.append(gradient)

grid_display(image_all, 4, (20,20), title = True)
img = cv2.imread('../input/malaria-parasite-image-malaria-species/Malariae/img/1401080976-0003-T.tif',0)



dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)



magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



plt.figure(figsize=(15,10))

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
rows, cols = img.shape

crow,ccol = rows/2 , cols/2



# create a mask first, center square is 1, remaining all zeros

mask = np.zeros((rows,cols,2),np.uint8)

mask[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1



# apply mask and inverse DFT

fshift = dft_shift*mask

f_ishift = np.fft.ifftshift(fshift)

img_back = cv2.idft(f_ishift)

img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])



plt.figure(figsize=(15,10))

plt.subplot(121),plt.imshow(img, cmap = 'gray')

plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(img_back, cmap = 'gray')

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()