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
import pandas as pd

import tensorflow as tf

import keras

import cv2

import os

import imageio

import numpy as np

from sklearn.model_selection import train_test_split

from keras import models

from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
train_dir = r'/kaggle/input/fruits/fruits-360_dataset/fruits-360/Training'

test_dir= r'/kaggle/input/fruits/fruits-360_dataset/fruits-360/Test'

train_classes={}

test_classes={}

i=0

for item in os.listdir(train_dir):

    cls = item.split(" ")[0]

    if cls not in train_classes:

#         print(i)

        train_classes[cls] = i

        i+=1
i=0

for item in os.listdir(test_dir):

    cls = item.split(" ")[0]

    if cls not in test_classes:

#         print(i)

        test_classes[cls] = i

        i+=1
x = []

y=[]



for item in os.listdir(train_dir):

    for image in os.listdir(train_dir+'/'+item):

        #print(image)

        img = imageio.imread(train_dir+'/'+item+'/'+image)

        x.append(img)

        cls = train_classes[item.split(" ")[0]]

        y.append(cls)



arr_x=np.array(x)

arr_y=np.array(y)

new_y = keras.utils.to_categorical(arr_y)

x_train, x_test, y_train, y_test = train_test_split(arr_x, new_y, test_size=0.25, random_state=42)
model = models.Sequential()




model.add(Conv2D(filters=32, kernel_size=(3,3), padding='SAME', input_shape=x_train[0].shape))



model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.3))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=256, kernel_size=(3,3), padding='SAME'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.3))



model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.2))



model.add(Conv2D(filters=64, kernel_size=(3,3), padding='SAME'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))

model.add(Dropout(rate=0.2))



model.add(Dense(63))

model.add(Activation('softmax'))

opt = keras.optimizers.Adam()

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=5)
loss, score = model.evaluate(x_test, y_test)
print(score)
test_x = []

test_y=[]



for item in os.listdir(test_dir):

    for image in os.listdir(test_dir+'/'+item):

        #print(image)

        img = imageio.imread(test_dir+'/'+item+'/'+image)

        test_x.append(img)

        cls = test_classes[item.split(" ")[0]]

        test_y.append(cls)



t_x=np.array(test_x)

t_y=np.array(test_y)

new_t_y = keras.utils.to_categorical(t_y)

t_loss, t_score = model.evaluate(t_x, new_t_y)
print(score)

print(loss)