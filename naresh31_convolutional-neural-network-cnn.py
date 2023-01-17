# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/violencedataset/ViolenceDataset"))



# Any results you write to the current directory are saved as output.
import keras
import tensorflow as tf
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing import image

from keras.utils.np_utils import to_categorical

import os
gun=[]

hand=[]



folderpath_hand = "../input/violencedataset/ViolenceDataset/hands/"

for filename in os.listdir(folderpath_hand):

	imagepath = folderpath_hand + filename

	img = image.load_img(imagepath, target_size=(150,150))

	hand.append((1/255)*np.asarray(img))

	

folderpath_hand = "../input/violencedataset/ViolenceDataset/hand_guns/"

for filename in os.listdir(folderpath_hand):

	imagepath = folderpath_hand + filename

	img = image.load_img(imagepath, target_size=(150,150))

	gun.append((1/255)*np.asarray(img))   
gun_train = np.random.randint(len(gun), size=int(0.8*len(gun)))

gun_test = set(range(len(gun))).difference(set(range(len(gun_train))))

hand_train = np.random.randint(len(hand), size=int(0.8*len(hand)))

hand_test = set(range(len(hand))).difference(set(range(len(hand_train))))
gun_train
for i in gun_train:

    gun_x_train.append(gun[i])

    gun_y_train.append(1)



for i in hand_train:

    hand_x_train.append(hand[i])

    hand_y_train.append(0)



for i in gun_test:

    gun_x_test.append(gun[i])

    gun_y_test.append(1)



for i in hand_test:

    hand_x_test.append(hand[i])

    hand_y_test.append(0)