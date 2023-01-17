# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Using the LetterColorImages2.h5 for training and LetterColorImages.h5 as an additional test set
import h5py
letters = h5py.File("../input/LetterColorImages2.h5", "r")
letters_first = h5py.File("../input/LetterColorImages.h5", "r")
list(letters.keys())
#Creating arrays for the images, labels contained in the LetterColorImages.h5 file
backgrounds_1 = np.array(letters_first['backgrounds'])
images_1 = np.array(letters_first['images'])
labels_1 = np.array(letters_first['labels'])
#Creating arrays for the images, labels contained in the LetterColorImages2.h5 file
backgrounds = np.array(letters['backgrounds'])
images = np.array(letters['images'])
labels = np.array(letters['labels'])
print('Have the background, images and labels...')
#Preview of some images
from matplotlib import pyplot as plt
for i in range(5):
    plt.imshow(images[i], interpolation='nearest')
    plt.show()
#Preview of some images
for i in range(5):
    plt.imshow(images[1000+i], interpolation='nearest')
    plt.show()
#Preview of some images
for i in range(5):
    plt.imshow(images[2000+i], interpolation='nearest')
    plt.show()
#Preview of some images - from the first dataset
for i in range(5):
    plt.imshow(images_1[i], interpolation='nearest')
    plt.show()
backgrounds.shape
backgrounds.dtype
images.shape
labels.shape
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils import to_categorical

new_labels = labels - 1
y = to_categorical(new_labels, 33)
print('One hot encoding of labels is complete...')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.25)
print('Training: %d Test: %d' %(X_train.shape[0], X_test.shape[0]))

train_mean = np.mean(X_train)
print('Mean prior to preprocessing: ', train_mean)
X_train -= train_mean
X_test -= train_mean
print('Mean after preprocessing for training set: ', np.mean(X_train))
print('Mean after preprocessing for test set: ', np.mean(X_test))
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, MaxPooling2D
print('Import complete')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(33, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, validation_split = 0.2)
#lets see how we did based on the test set kept aside 
model.evaluate(X_test, y_test)
#One hot encoding of the first dataset labels
new_labels_1 = labels_1 - 1
y1 = to_categorical(new_labels_1, 33)
#Lets see how well we do with the first dataset
images_1_centered = images_1 - train_mean
model.evaluate(images_1_centered, y1)