import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2

import albumentations

from albumentations import torch as AT

import os

print(os.listdir("../input"))
from albumentations import (

    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,

    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,

    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,

    IAASharpen, IAAEmboss, Flip, OneOf, Compose

)
base_dr = "../input/"

# Reading the CSVs

train = pd.read_csv(base_dr+'train.csv')

test = pd.read_csv(base_dr+'test.csv')
def strong_aug(p=.5):

    # source - https://github.com/albu/albumentations/blob/master/notebooks/example.ipynb

    return Compose([

        RandomRotate90(),

        Flip(),

        Transpose(),

        OneOf([

            IAAAdditiveGaussianNoise(),

            GaussNoise(),

        ], p=0.2),

        OneOf([

            MotionBlur(p=.2),

            MedianBlur(blur_limit=3, p=0.1),

            Blur(blur_limit=3, p=0.1),

        ], p=0.2),

        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),

        OneOf([

            OpticalDistortion(p=0.3),

            GridDistortion(p=.1),

            IAAPiecewiseAffine(p=0.3),

        ], p=0.2),

        OneOf([

            CLAHE(clip_limit=2),

            IAASharpen(),

            IAAEmboss(),

            RandomBrightnessContrast(),            

        ], p=0.3),

        HueSaturationValue(p=0.3),

    ], p=p)
x = [] ; y = [] ; names = []

for id in (train.id_code.values):

    full_size_image = cv2.imread('../input/train_images/{}.png'.format(id))

    imgFile = cv2.resize(full_size_image, (224,224) , interpolation=cv2.INTER_CUBIC)

    aug = HorizontalFlip(always_apply=True) # horization flip

    image = aug(image=imgFile)['image']

    names.append('{}_HP'.format(id))

    x.append(image)

    y.append(train[train['id_code'] == id].diagnosis.values[0])

    aug = ShiftScaleRotate(always_apply=True) # Random Shift Scale & Rotate

    image = aug(image=imgFile)['image']

    names.append('{}_SSR'.format(id))

    x.append(image)

    y.append(train[train['id_code'] == id].diagnosis.values[0])

    aug = strong_aug(p=1) # Strong Random augenmation from above function

    image = aug(image=imgFile)['image']

    names.append('{}_StrAug'.format(id))

    x.append(image)

    y.append(train[train['id_code'] == id].diagnosis.values[0])
##??ShiftScaleRotate
# Set it up as a dataframe if you like

df = pd.DataFrame() ; df["labels"]=y ; df["images"]=x ; df['names'] = names
#Looking at shapes and first image

df.shape , df.images[0].shape , df.labels[0]
# Looking at augmented data

df.head()
from random import sample

import cv2

full_size_image = cv2.imread('../input/train_images/000c1434d8d7.png')

imgFile = cv2.resize(full_size_image, (224,224) , interpolation=cv2.INTER_CUBIC)

plt.imshow(full_size_image)

imgFile.shape
print(df[df['names'].str.contains("000c1434d8d7")]['names'][0])

plt.imshow(df[df['names'].str.contains("000c1434d8d7")]['images'][0]);
print(df[df['names'].str.contains("000c1434d8d7")]['names'][1])

plt.imshow(df[df['names'].str.contains("000c1434d8d7")]['images'][1]);



## this is not working , SSR - need to work more
print(df[df['names'].str.contains("000c1434d8d7")]['names'][2])

plt.imshow(df[df['names'].str.contains("000c1434d8d7")]['images'][2]);
# Work in progress