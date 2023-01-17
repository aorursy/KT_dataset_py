# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
train_img_path = "/kaggle/input/understanding_cloud_organization/train_images/"

test_img_path = "/kaggle/input/understanding_cloud_organization/test_images/"

## 1.a

print("1.a How many images does the dataset consist of?")



print(f"\tNumber of images in the train set is {len(os.listdir(train_img_path))}")

print(f"\tNumber of images in the test set is {len(os.listdir(test_img_path))}")



## 1.b

print("\n1.b How many classes? How many images per class?")

train = pd.read_csv("/kaggle/input/understanding_cloud_organization/train.csv")

labels_times = {"Fish":0,"Flower":0,"Gravel":0,"Sugar":0}

img_per_label = [0,0,0,0]

for name in labels_times:

    for p in range(train.shape[0]):

        if name in train["Image_Label"][p] and pd.isna(train["EncodedPixels"][p]) == False:

            labels_times[name] +=1

    print(f"\t{name}:\t{labels_times[name]}")



## 1.c

print("1.c Show 4 samples from each class.")

print("\t**The pictures can have more than one class")

print()

print()

for name in labels_times:

    i = 0

    print(f"\tClass: {name}")

    plt.figure(1)

    for p in range(train.shape[0]):

        if name in train["Image_Label"][p] and pd.isna(train["EncodedPixels"][p]) == False and i <4:

            i += 1

            plt.subplot(140+i)

            plt.imshow(cv2.imread(train_img_path +train["Image_Label"][p].split("_")[0]))

    plt.show()
