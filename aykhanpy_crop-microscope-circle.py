import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2
def crop_microscope(img_to_crop):

    pad_y = img_to_crop.shape[0]//200 

    pad_x = img_to_crop.shape[1]//200

    img = img_to_crop[pad_y:-pad_y, pad_y:-pad_y,:]

    '''

cropping 0.5% from every side, because some microscope images

have frames along the edges so cv2.boundingRect crops by frame, 

but not by needed part of the image.

    '''

    

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY) 

    x,y,w,h = cv2.boundingRect(thresh) #getting crop points

    

#since we cropped borders we need to uncrop it back 

    if y!=0: 

        y = y+pad_y

    if h == thresh.shape[0]:

        h = h+pad_y

    if x !=0:

        x = x +pad_x

    if w == thresh.shape[1]:

        w = w + pad_x

    h = h+pad_y

    w = w + pad_x

    crop = img_to_crop[y:y+h,x:x+w]

    return crop
#some examples

pathes = [

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0073502.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0170285.jpg',

    '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_0072042.jpg',

    '../input/melanoma-merged-external-data-512x512-jpeg/512x512-dataset-melanoma/512x512-dataset-melanoma/ISIC_0063521.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0197440.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0112420.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0336093.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0371907.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0591142.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_0874437.jpg',

    '../input/siim-isic-melanoma-classification/jpeg/test/ISIC_1089351.jpg',

]
plt.figure(figsize = (20,70))

for i in range(len(pathes)):

    img = cv2.imread(pathes[i])

    plt.subplot(len(pathes), 2, i*2+1)

    plt.imshow(crop_microscope(img))

    plt.title(f'Cropped shape: {crop_microscope(img).shape}', fontsize=20)

    plt.subplot(len(pathes), 2, i*2+2)

    plt.imshow(img)

    plt.title(f'Original shape: {img.shape}', fontsize=20);