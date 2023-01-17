# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from PIL import Image

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Flatten, Dense, Dropout, Activation
infected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/') 

uninfected = os.listdir('../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/')
data = []

labels = []



for i in infected:

    try:

    

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized/"+i)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated45 = resize_img.rotate(45)

        rotated75 = resize_img.rotate(75)

        blur = cv2.blur(np.array(resize_img) ,(10,10))

        data.append(np.array(resize_img))

        data.append(np.array(rotated45))

        data.append(np.array(rotated75))

        data.append(np.array(blur))

        labels.append(1)

        labels.append(1)

        labels.append(1)

        labels.append(1)

        

    except AttributeError:

        print('')

        

for u in uninfected:

    try:

        

        image = cv2.imread("../input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected/"+u)

        image_array = Image.fromarray(image , 'RGB')

        resize_img = image_array.resize((50 , 50))

        rotated45 = resize_img.rotate(45)

        rotated75 = resize_img.rotate(75)

        data.append(np.array(resize_img))

        data.append(np.array(rotated45))

        data.append(np.array(rotated75))

        labels.append(0)

        labels.append(0)

        labels.append(0)

        

    except AttributeError:

        print('')
cells = np.array(data)

labels = np.array(labels)
cells.shape, labels.shape
n = np.arange(cells.shape[0])

np.random.shuffle(n)

cells = cells[n]

labels = labels[n]

labels[2]
cells = cells.astype(np.float32)

labels = labels.astype(np.float32)

cells = cells / 255.0
from sklearn.model_selection import train_test_split



train_x, test_x, train_y, test_y = train_test_split(cells, labels, test_size=0.2, random_state=1)
train_x.shape, test_x.shape
model = Sequential()

model.add(Conv2D(64,(7,7), padding="same", activation="relu"))

model.add(Conv2D(32,(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D((2,2)))

model.add(Conv2D(16,(3,3), padding="same", activation="relu"))



model.add(Flatten())

model.add(Dense(512, activation="relu"))

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(2, activation="softmax"))
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)



model.compile(loss='sparse_categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
model.fit(train_x, train_y,

              batch_size=128,

              epochs=20,

              validation_data=(test_x, test_y))