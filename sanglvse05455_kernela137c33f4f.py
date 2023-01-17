import numpy as np

import matplotlib.pyplot as plt

import keras

from keras.layers import Dense, Flatten, Conv2D, MaxPool2D

from keras.models import Sequential

from keras import metrics 

from keras.optimizers import SGD

import os

import cv2

import random

from PIL import Image

import imgaug.augmenters as iaa
DIR = '../input/catndog/catndog'

LABELS = ["cat", "dog"]



x_train = []

y_train = []



for label in LABELS:

    path = os.path.join(DIR + '/train', label)

    class_num = LABELS.index(label)

    for img_path in os.listdir(path):

        img = cv2.imread(os.path.join(path, img_path))

        new_img = cv2.resize(img, (32, 32))

        x_train.append(new_img)

        y_train.append(class_num)

        

x_test = []

y_test = []



for label in LABELS:

    path = os.path.join(DIR + '/test', label)

    class_num = LABELS.index(label)

    for img_path in os.listdir(path):

        img = cv2.imread(os.path.join(path, img_path))

        new_img = cv2.resize(img, (32, 32))

        x_test.append(new_img)

        y_test.append(class_num)

for i in range(0, 9):

    r = random.randint(0, 152)

    ax = plt.subplot(3, 3, i+1)

    ax = plt.imshow(x_train[r])

    plt.title(LABELS[y_train[r]])
seq = iaa.Sequential([

    #iaa.Crop(percent=0.5), 

    iaa.CropAndPad(

        percent=(0.5),

    ),

    iaa.Fliplr(0.7),

    iaa.MedianBlur(k=3),

    iaa.Affine(rotate=(-45, 45))

])



x_train_aug = seq.augment_images(x_train)
for i in range(0, 9):

    r = random.randint(0, 152)

    ax = plt.subplot(3, 3, i+1)

    ax = plt.imshow(x_train_aug[r])

    plt.title(LABELS[y_train[r]])