import numpy as np

import pandas as pd

import json

import sys

from skimage.io import imread

from matplotlib import pyplot as plt

from keras import models

from keras.optimizers import SGD

import keras.models as models

from keras.layers import add

from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute

from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization

from keras import backend as K

import cv2



from tqdm import tqdm

import os

import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

input_shape = (224, 224, 3)

from keras.applications import VGG19

vgg_con = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))



from keras.layers import UpSampling2D



vgg_con.summary()
model = models.Sequential()

model.add(vgg_con)
model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(UpSampling2D((2, 2)))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))

model.add(Conv2D(1, (3, 3), activation="relu", padding="same"))
model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile( loss = "sparse_categorical_crossentropy", 

               optimizer = sgd, 

               metrics=['accuracy']

             )
model.fit(X, y, epochs=150, validation_split=0.2)

names = []
data = []

label = []

for i in tqdm(os.listdir("../input/first_part")):

    sub_folder_path = os.path.join("../input/first_part", i)

    for j in tqdm(os.listdir(sub_folder_path)):

        sub_sub_folder = os.path.join(sub_folder_path, j)

        for ij in os.listdir(sub_sub_folder):

            if ".jpg" in ij:

                img_name = ij.split('.jpg')[0]

                names.append([img_name, sub_sub_folder])
names = np.array(names)
names[0]
X = []

y = []

for img_info in tqdm(names):

    name = img_info[0]

    orignal_path = img_info[1] + "/" + name + ".jpg"

    seg_path = img_info[1] + "/" + name + "_seg.png"

    img = cv2.imread(orignal_path)

    img = cv2.resize(img, (224, 224))

    seg = cv2.imread(seg_path, 0)

    seg = cv2.resize(seg, (224, 224))

    X.append(img)

    y.append(seg)
print("done")
X = np.array(X)

y = np.array(y)
X.shape

y = y.reshape((-1, 224, 224, 1))
y.shape
name = img_info[0]

name
X_ = []

y_ = []

for img_info in tqdm(names[2000:2050]):

    name = img_info[0]

    orignal_path = img_info[1] + "/" + name + ".jpg"

    seg_path = img_info[1] + "/" + name + "_seg.png"

    img = cv2.imread(orignal_path)

    img = cv2.resize(img, (224, 224))

    print(img.shape)

    seg = cv2.imread(seg_path)

    seg = cv2.resize(seg, (224, 224))

    print(seg.shape)

    X_.append(img)

    y_.append(seg)
plt.imshow(img)
plt.imshow(seg)
img = img.reshape(1, 224, 224, 3)
X_ = np.array(X_)
X_.shape
output = model.predict(X_)
output[0]*255
print(output.shape[0])
for img in range(output.shape[0]):

    img1 = output[img]

    plt.imshow((img1 * 255).astype(np.uint8))

    plt.show()
output = output.reshape(224, 224, 3)
output.shape
plt.imshow(output)