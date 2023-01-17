# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import cv2

import matplotlib.pyplot as plt

direc = "/kaggle/input/cats-and-dogs-sentdex-tutorial/PetImages/"

categories = ['Dog', 'Cat']

for category in categories:

    path = os.path.join(direc, category)

    for img in os.listdir(path):

        img_read = cv2.imread(os.path.join(path, img))

        new_read = cv2.resize(img_read, (100, 100))

        plt.imshow(new_read)

        break

    break

# Any results you write to the current directory are saved as output.
#load training_data

training_data = []

SIZE = 100   #fix the size for all images

for category in categories:

    path = os.path.join(direc, category)

    class_num = categories.index(category)

    for img in os.listdir(path):

        try:

            img_read = cv2.imread(os.path.join(path, img))

            new_read = cv2.resize(img_read, (SIZE, SIZE))

            training_data.append([new_read, class_num])

        except Exception as e:

            pass



print(len(training_data))
#Now shuffle the training data

import random

random.shuffle(training_data)
x = []

y = []

for features, label in training_data:

    x.append(features)

    y.append(label)



print(str(len(x)) + "  |  " + str(len(y)))
x = np.array(x).reshape(-1, SIZE, SIZE, 3)

print(x.shape)

y = np.array(y)

print(y.shape)

X = x/255.0

plt.imshow(X[0])
import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.layers import BatchNormalization, Dropout

print('using Tensorflow with Keras')
model = Sequential()

model.add(Conv2D(64, (3,3), kernel_initializer = 'he_normal', input_shape = X.shape[1:]))

model.add(BatchNormalization(axis = -1))

model.add(Activation('relu'))

model.add(MaxPooling2D((3,3), strides = (2,2)))

model.add(Dropout(rate = 0.5))



model.add(Conv2D(128, (3,3), kernel_initializer = 'he_normal'))

model.add(BatchNormalization(axis = -1))

model.add(Activation('relu'))

model.add(MaxPooling2D((3,3), strides = (2,2)))

model.add(Dropout(rate = 0.5))



model.add(Conv2D(256, (3,3), kernel_initializer = 'he_normal'))

model.add(BatchNormalization(axis = -1))

model.add(Activation('relu'))

model.add(MaxPooling2D((3,3), strides = (2,2)))

model.add(Dropout(rate = 0.5))



model.add(Conv2D(512, (3,3), kernel_initializer = 'he_normal'))

model.add(BatchNormalization(axis = -1))

model.add(Activation('relu'))

model.add(MaxPooling2D((3,3), strides = (2,2)))

model.add(Dropout(rate = 0.5))



model.add(Flatten())

model.add(Dense(1024, use_bias = True))

model.add(Activation('relu'))

model.add(Dropout(rate = 0.5))



model.add(Dense(512, use_bias = True))

model.add(Activation('relu'))



model.add(Dense(128, use_bias = True))

model.add(Activation('relu'))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



history = model.fit(X, y, batch_size = 64, epochs = 50, validation_split = 0.1)
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
#test dataset

test_data = []

direc = "/kaggle/input/cat-and-dog/test_set/test_set"

categories = ['dogs', 'cats']

SIZE = 100   #fix the size for all images

for category in categories:

    path = os.path.join(direc, category)

    class_num = categories.index(category)

    for img in os.listdir(path):

        try:

            img_read = cv2.imread(os.path.join(path, img))

            new_read = cv2.resize(img_read, (SIZE, SIZE))

            test_data.append([new_read, class_num])

        except Exception as e:

            pass



print(len(test_data))

    
random.shuffle(test_data)

test_x = []

test_y = []

for features, label in test_data:

    test_x.append(features)

    test_y.append(label)

test_x = np.array(test_x).reshape(-1, SIZE, SIZE, 3)

print(test_x.shape)

test_y = np.array(test_y)

print(test_y.shape)

test_x = test_x/255.0

plt.imshow(test_x[0])
val_loss, val_accu = model.evaluate(test_x, test_y)

print(val_loss)

print(val_accu)
model.summary()
model.save("/kaggle/working/catvsdog.h5")

print('Saved Successfully')