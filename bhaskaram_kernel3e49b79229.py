# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing the needed libraries

import numpy as np

import matplotlib.pyplot as plt

import os

import cv2
 # loading the

TRAIN_DATA = '/kaggle/input/dogs-vs-cats/train/train'



# go through all the examples of the images

examples = { 'dog': [], 'cat': [] }



for dirname,_, filenames in os.walk(TRAIN_DATA):

    for filename in filenames:

        if 'dog' in filename:

            examples['dog'].append(cv2.imread(os.path.join(dirname, filename), 0))

        else:

            examples['cat'].append(cv2.imread(os.path.join(dirname, filename), 0))

print(len(examples['dog']))

print(len(examples['cat']))
sample_cat = examples['cat'][0]

plt.imshow(sample_cat, cmap='gray')

plt.show()
IMG_SIZE = 50

plt.imshow(cv2.resize(sample_cat, (IMG_SIZE, IMG_SIZE)), cmap='gray')

plt.show()
def create_training_data():

    TRAIN_DATA = '/kaggle/input/dogs-vs-cats/train/train'

    # array for the sampel data

    training_data = []

    # walk through all the images and laod the training data

    for dirname,_, filenames in os.walk(TRAIN_DATA):

        for filename in filenames:

            category = 0 if 'dog' in filename else 1 

            img = cv2.imread(os.path.join(dirname, filename), 0)

            resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            training_data.append([resized_img, category])

    return training_data
training_data = create_training_data()
X = []

y = []

for feature, label in training_data:

    X.append(feature)

    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE,IMG_SIZE, 1)

print(len(X))

print(len(y))
import pickle



pickle_out = open('X.pickle', 'wb')

pickle.dump(X, pickle_out)

pickle_out.close()





pickle_out = open('y.pickle', 'wb')

pickle.dump(y, pickle_out)

pickle_out.close()



pickle_in = open('X.pickle', 'rb')

some_X = pickle.load(pickle_in)
# loading some values here
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D 
train_x = X / 255.0
model = Sequential()



model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Conv2D(64, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))



model.add(Flatten())

model.add(Dense(64))



model.add(Dense(1))

model.add(Activation('sigmoid'))

          

          

model.compile(loss='binary_crossentropy',

             optimizer = 'adam',

             metrics = ['accuracy'])

          

model.fit(train_x, y, batch_size = 32, epochs = 3, validation_split = 0.1)