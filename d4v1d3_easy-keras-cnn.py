# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir("/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images")



# Any results you write to the current directory are saved as output.
uninfected = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Uninfected"

infected = "/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images/Parasitized"
import cv2

# get the images and the labels

imgs = []

labels = []

for file in os.listdir(uninfected)[:]:

    imgs.append(cv2.imread(uninfected + "/" + file))

    labels.append(1)



for file in os.listdir(infected)[:]:

    imgs.append(cv2.imread(infected + "/" + file))

    labels.append(0)
# resize the images to 160x151

resized = []

res_labels = []

for n in range(len(imgs)):

    i = imgs[n]

    l = labels[n]

    try:

        resized.append(cv2.resize(i, (160, 151), interpolation = cv2.INTER_AREA))

        res_labels.append(l)

    except:

        print("error")

    
# split the dataset

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(resized, res_labels, test_size=0.33, random_state=42)

# change the shape for CNN

X_train = np.asarray(X_train).reshape(-1,151, 160, 3)  

X_test = np.asarray(X_test).reshape(-1,151, 160, 3)
# create the model

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers



model = keras.Sequential()



model.add(layers.Conv2D(16, kernel_size=3, activation=tf.nn.selu, input_shape=(151,160,3)))

model.add(layers.Dropout(.1))

model.add(layers.MaxPooling2D(2))

model.add(layers.Conv2D(64, kernel_size=3, activation=tf.nn.selu, input_shape=(151,160,3)))

model.add(layers.Dropout(.1))

model.add(layers.MaxPooling2D(2))

model.add(layers.Flatten())

model.add(layers.Dense(200, activation=tf.nn.selu))

model.add(layers.Dense(1, activation=tf.nn.sigmoid))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
# fitting the model

model.fit(X_train, y_train, batch_size=500, epochs=30)

model.evaluate(X_test, y_test)