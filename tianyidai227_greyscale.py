import os

import numpy as np

import cv2

import matplotlib.pyplot as plt

import pandas as pd
base_path = '/kaggle/input/siim-isic-melanoma-classification'

train = pd.read_csv(os.path.join(base_path, 'train.csv'))

len(train)
newpath = r'/kaggle/working/grayscale' 

if not os.path.exists(newpath):

    os.makedirs(newpath)
save_path = '/kaggle/working/grayscale'

for i in range(0,len(train)):

    image = cv2.imread(base_path + '/jpeg/train/' + train['image_name'][i] + '.jpg')

    image_rieze = cv2.resize(image, (200,200))

    final_image = cv2.cvtColor(image_rieze, cv2.COLOR_RGB2GRAY)

    completeName = os.path.join(save_path, train['image_name'][i]+'.jpg') 

    cv2.imwrite(completeName, final_image)