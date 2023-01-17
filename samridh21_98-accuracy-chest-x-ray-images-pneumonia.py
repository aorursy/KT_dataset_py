import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

import random

import os

import cv2

import numpy as np

train_folder = '../input/chest-xray-pneumonia/chest_xray/train'

test_folder = '../input/chest-xray-pneumonia/chest_xray/test'

val_folder = '../input/chest-xray-pneumonia/chest_xray/val'

labels = ["NORMAL", "PNEUMONIA"] # each folder has two sub folder name "PNEUMONIA", "NORMAL"

IMG_SIZE = 70 # resize image

def get_data_train(data_dir):

    data = []

    for label in labels:

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for img in os.listdir(path):

            try:

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                data.append([new_array, class_num])

            except Exception as e:

                print(e)

    return np.array(data)
train=get_data_train(train_folder)

test = get_data_train(test_folder)
X_train = []

y_train = []



X_val = []

y_val = []



X_test = []

y_test = []



for feature, label in train:

    X_train.append(feature)

    y_train.append(label)



for feature, label in test:

    X_test.append(feature)

    y_test.append(label)
X_train = np.array(X_train) / 255

X_test = np.array(X_test) / 255

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = np.array(y_train)

X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_test = np.array(y_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation,Dropout,Conv2D,Dense,Flatten,MaxPooling2D
model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=X_train.shape[1:]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,validation_split=0.1,batch_size=32)

model.save('CNN_Pneumonia.model')