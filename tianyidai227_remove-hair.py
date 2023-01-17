import os

import numpy as np

import cv2

import matplotlib.pyplot as plt

import pandas as pd
base_path = '/kaggle/input/siim-isic-melanoma-classification'

save_path = '/kaggle/working/remove_hair'

train = pd.read_csv(os.path.join(base_path, 'train.csv'))

len(train)
def hair_remove(image):

    # convert image to grayScale

    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    

    # kernel for morphologyEx

    kernel = cv2.getStructuringElement(1,(17,17))

    

    # apply MORPH_BLACKHAT to grayScale image

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    

    # apply thresholding to blackhat

    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

    

    # inpaint with original image and threshold image

    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)

    

    return final_image
newpath = r'/kaggle/working/remove_hair' 

if not os.path.exists(newpath):

    os.makedirs(newpath)
for i in range(0,len(train)):

    image = cv2.imread(base_path + '/jpeg/train/' + train['image_name'][i] + '.jpg')

    image_resize = cv2.resize(image,(256,256))

    final_image = hair_remove(image_resize)

    completeName = os.path.join(save_path, train['image_name'][i]+'.jpg') 

    cv2.imwrite(completeName, final_image)