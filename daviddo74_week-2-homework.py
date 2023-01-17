# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
DATA_DIR_TRAIN = "../input/catndog/catndog/train"

CATEGORIES = ["cat", "dog"]

x_train = []

y_train = []

IMG_SIZE = 128

for category in CATEGORIES:

    path = os.path.join(DATA_DIR_TRAIN, category)

    class_num = CATEGORIES.index(category)

    for img_path in os.listdir(path):

        img = cv2.imread(os.path.join(path, img_path))

        new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        x_train.append(new_img)

        y_train.append(class_num)
x_train = np.array(x_train)

y_train = np.array(y_train)
random_array = np.random.randint(len(x_train),size=9)

random_array
grids = (3,3)

counter = 0



plt.figure(figsize=(10,10))



for i in random_array:

    img = x_train[i]

    label = y_train[i]

    

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter)

    ax = plt.imshow(img, cmap='brg')

    plt.xticks([])

    plt.yticks([])

    plt.title(CATEGORIES[int(label)])
import imgaug.augmenters as iaa
seq = iaa.Sequential([

    #iaa.Crop(percent=0.5), 

    iaa.CropAndPad(

        percent=(0.5),

    ),

    iaa.Fliplr(0.7),

    iaa.MedianBlur(k=3),

    iaa.Affine(rotate=(-45, 45))

])
images_aug = seq.augment_images(x_train)
grids = (3,3)

counter = 0



plt.figure(figsize=(10,10))



for i in random_array:

    img = images_aug[i]

    label = y_train[i]

    

    if(counter < grids[0]*grids[1]):

        counter += 1

    else:

        break

    

    # plot image and its label

    ax = plt.subplot(grids[0], grids[1], counter)

    ax = plt.imshow(img, cmap='brg')

    plt.xticks([])

    plt.yticks([])

    plt.title(CATEGORIES[int(label)])