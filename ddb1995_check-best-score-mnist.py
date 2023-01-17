# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# Converting the Labels to 10 different classes

Y_train = train['label']

Y_train = to_categorical(Y_train, num_classes=10)
# Preparing the data for X_train and reshaping it to (-1, 28, 28, 1) -> where -1 represents new shape should be compatible with original shape and

# 1 represents the channel 

# 28 and 28 represents the height and width of the resized images

# Dividing by 255 will normalize the values between 0 and 1

X_train = train.iloc[:,1:]

X_train = X_train/255

X_train = X_train.values.reshape(-1, 28, 28, 1)
# Preparing the data for X_test



X_test = test/255

X_test = X_test.values.reshape(-1,28,28,1)
print(X_train.shape)

print(X_test.shape)
plt.figure(figsize=(15,5))

for i in range(30):

    plt.subplot(3,10, i+1)

    plt.axis('off')

    plt.imshow(X_train[i].reshape(28,28))
datagenerate = ImageDataGenerator(rotation_range=10, zoom_range=0.10, width_shift_range=0.1, height_shift_range=0.1)
X_train3 = X_train[9,].reshape((-1,28,28,1))

Y_train3 = Y_train[9,].reshape((1,10))

plt.figure(figsize=(15,4.5))



for i in range(30):

    plt.subplot(3,10,i+1)

    xt2, yt2 = datagenerate.flow(X_train3, Y_train3).next()

    plt.imshow(xt2[0].reshape((28,28)))

    plt.axis('off')

    if i==9:

        X_train3 = X_train[32,].reshape(-1, 28, 28, 1)

    if i==19:

        X_train3 = X_train[33,].reshape(-1, 28, 28, 1)

nets = 3

model = [0] *nets



for j in range(3):

    model[j] = Sequential()

    model[j].add(Conv2D(24,kernel_size=5,padding='same',activation='relu',

            input_shape=(28,28,1)))

    model[j].add(MaxPool2D())

    if j>0:

        model[j].add(Conv2D(48,kernel_size=5,padding='same',activation='relu'))

        model[j].add(MaxPool2D())

    if j>1:

        model[j].add(Conv2D(64,kernel_size=5,padding='same',activation='relu'))

        model[j].add(MaxPool2D(padding='same'))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation='relu'))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# Validation Set

X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.333)

# Train Networks

history = [0] * nets

names = ["(C-P)x1","(C-P)x2","(C-P)x3"]
