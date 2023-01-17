# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt

import os
train_data = pd.read_csv('/kaggle/input/train.csv')

train_data.head()
img = cv2.imread('/kaggle/input/train/train/0014d7a11e90b62848904c1418fc8cf2.jpg')

print(img.shape)

plt.imshow(img)
print(train_data['id'][0])

a = []

a.append(int(train_data.loc[train_data['id'] == '0014d7a11e90b62848904c1418fc8cf2.jpg']['has_cactus']))

a
train_path = '/kaggle/input/train/train/'

X_train = []

y_train = []

train_images = os.listdir(train_path)



for i in range(len(train_images)):

    img = cv2.imread(train_path+train_images[i]) #read all images

    img = img/255

    X_train.append(img) # append to list

    y_train.append(int(train_data.loc[train_data['id'] == str(train_images[i])]['has_cactus']))

print(len(X_train))

print(len(y_train))
X_train = np.array(X_train)

y_train = np.array(y_train)
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
input_shape = (32, 32, 3)



model = Sequential()



model.add(Conv2D(32, 2, 2, padding='same', input_shape=input_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, 2, 2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())



model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1))

model.add(Activation('sigmoid'))

    

model.compile(loss='binary_crossentropy',

            optimizer=RMSprop(lr=0.0001),

            metrics=['accuracy'])
model.fit(x=X_train, y= y_train, batch_size=64, epochs=80, validation_split=0.2, shuffle=True)
sample_sub = pd.read_csv('/kaggle/input/sample_submission.csv')

sample_sub.head()
sample_sub.has_cactus.unique()
test_path = '/kaggle/input/test/test/'

X_test = {}

test_images = os.listdir(test_path)



for i in range(len(test_images)):

    img = cv2.imread(test_path+test_images[i]) #read all images

    img = img/255

    img = img.reshape(1, 32, 32, 3)

    prediction = model.predict(img)

    prediction = prediction[0][0]

    if prediction>0.5:

        prediction = 1

    else:

        prediction = 0

    X_test[test_images[i]] = prediction

# print(X_test)

test_df = pd.DataFrame(list(X_test.items()), columns=['id', 'has_cactus'])

test_df.head()
test_df.to_csv("cactus_prediction.csv", index=False)