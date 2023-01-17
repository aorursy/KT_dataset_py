# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from PIL import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/cell_images/cell_images/"))

# Any results you write to the current directory are saved as output.
data=[]

labels=[]

Parasitized=os.listdir("../input/cell_images/cell_images/Parasitized/")

for i in Parasitized:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(1)

    except:

        continue

Uninfected=os.listdir("../input/cell_images/cell_images/Uninfected/")

for j in Uninfected:

    try:

        image=cv2.imread("../input/cell_images/cell_images/Uninfected/"+j)

        image_from_array = Image.fromarray(image, 'RGB')

        size_image = image_from_array.resize((50, 50))

        data.append(np.array(size_image))

        labels.append(0)

    except:

        continue

data=np.array(data)

labels=np.array(labels)

np.save("data",data)

np.save("labels",labels)
data=np.load("data.npy")

labels=np.load("labels.npy")

#data = data / 255

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle = True, test_size=0.30)

import keras as K

num_classes=len(np.unique(labels))

y_encoded_train=K.utils.to_categorical(y_train,num_classes)

y_encoded_test=K.utils.to_categorical(y_test,num_classes)
model = K.models.Sequential(

 [ K.layers.Conv2D(filters=128,kernel_size=3,padding="same",activation="relu",input_shape=(x_train.shape[1], x_train.shape[2],3)),

   K.layers.Conv2D(filters=64,kernel_size=3,padding="same",activation="relu"),

   K.layers.Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"),

   K.layers.Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"),

   K.layers.Conv2D(filters=16,kernel_size=2,padding="same",activation="relu"),

   K.layers.Flatten(),

   K.layers.Dense(256, activation='relu'),

   K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

   K.layers.Dense(128, activation="relu", kernel_initializer='random_uniform',bias_initializer='ones'),

   K.layers.Dropout(0.5),

   K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

   K.layers.Dense(64, activation="relu", kernel_initializer='random_uniform',bias_initializer='ones'),

   K.layers.Dropout(0.5),

   K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

   K.layers.Dense(32, activation="relu", kernel_initializer='random_uniform',bias_initializer='ones'),

   K.layers.Dropout(0.5),

   K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

   K.layers.Dense(16, activation="relu", kernel_initializer='random_uniform',bias_initializer='ones'),

   K.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'),

   K.layers.Dense(8, activation="relu", kernel_initializer='random_uniform',bias_initializer='ones'),

   K.layers.Dense(num_classes, activation='softmax')

 ]   

)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
model.fit(x_train,y_train, epochs =20, batch_size = 256)