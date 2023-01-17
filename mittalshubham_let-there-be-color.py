from keras.models import Model

from keras.layers import *

from keras.preprocessing.image import ImageDataGenerator

from keras import backend

from keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np

np.random.seed(40)

import os

import pandas as pd

import skimage.io, skimage.color, skimage.transform

from sklearn.model_selection import train_test_split

import tensorflow as tf
path = "../input/images256/"

files = np.array(pd.read_csv(path + "files.csv"))

files_train, files_valid, cls_train, cls_valid = train_test_split(files[:, 0], files[:, 1], test_size = 0.2, random_state = 40)

print(files_train.shape, files_valid.shape)

cls_train = to_categorical(cls_train, 205, dtype='int')

cls_valid = to_categorical(cls_valid, 205, dtype='int')
def batch_generator(for_train, batch_size):

    while True:

        if for_train is True:

            idx = np.random.randint(0, files_train.shape[0], batch_size)

        else:

            idx = np.random.randint(0, files_valid.shape[0], batch_size)

            

        gray = np.zeros((batch_size, 256, 256, 1))

        colo = np.zeros((batch_size, 256, 256, 2))

        cls = np.zeros((batch_size, 205))

        for i in range(batch_size):

            if for_train is True:

                temp = skimage.io.imread(path + files_train[idx[i]])

#                 print(temp.shape)

                cls[i,:] = cls_train[idx[i]]

            else:

                temp = skimage.io.imread(path + files_valid[idx[i]])

#                 print(temp.shape)

                cls[i,:] = cls_valid[idx[i]]

                

            temp = skimage.color.rgb2lab(temp)

            temp = skimage.transform.resize(temp, (256, 256))

            gray[i,:,:,0] = temp[:,:,0]

            colo[i] = temp[:,:,1:]

            

        gray = gray/100

        colo = (colo+128)/255

        yield gray, [cls, colo]
gray_image = Input(shape=(256, 256, 1))
# Low-level

conv = Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='tanh')

low_mid = conv(gray_image)

low_glo = conv(gray_image)



conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='tanh')

low_mid = conv(low_mid)

low_glo = conv(low_glo)



conv = Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='tanh')

low_mid = conv(low_mid)

low_glo = conv(low_glo)



conv = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='tanh')

low_mid = conv(low_mid)

low_glo = conv(low_glo)



conv = Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='tanh')

low_mid = conv(low_mid)

low_glo = conv(low_glo)



conv = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='tanh')

low_mid = conv(low_mid)

low_glo = conv(low_glo)
# Global-level

glo = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='tanh')(low_glo)

glo = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='tanh')(glo)

glo = Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='tanh')(glo)

glo = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='tanh')(glo)

glo = Flatten()(glo)

glo = Dense(1024, activation='tanh')(glo)

glo_512 = Dense(512, activation='tanh')(glo)

glo = Dense(256, activation='tanh')(glo_512)

print(glo)
# Classification

cls = Dense(256, activation='tanh')(glo_512)

cls = Dense(205, activation='softmax')(cls)
# Mid-level

mid = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='tanh')(low_mid)

mid = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='tanh')(mid)

print(mid)
# Fusion

print(mid)

print(glo)



fused = Reshape((1, 1, 256))(glo)

fused = Lambda(backend.tile, arguments={'n':(1, 32, 32, 1)})(fused)

fused = concatenate([mid, fused], 3)

fused = Dense(256, activation='tanh')(fused)

print(fused)
colo = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='tanh')(fused)

colo = UpSampling2D((2, 2))(colo)

colo = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='tanh')(colo)

colo = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='tanh')(colo)

colo = UpSampling2D((2, 2))(colo)

colo = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='tanh')(colo)

output = Conv2D(2, (3, 3), strides=(1, 1), padding='same', activation='sigmoid')(colo)

output = UpSampling2D((2, 2))(output)

print(output)
model = Model(inputs=gray_image, outputs=[cls, output])

model.compile(optimizer = "adam", loss = ["categorical_crossentropy", "mse"])

print(model.summary())
batch_size = 128

train_gen = batch_generator(True, batch_size = batch_size)

valid_gen = batch_generator(False, batch_size = batch_size)
model.load_weights("../input/let-there-be-color/weights.h5")
model.fit_generator(generator=train_gen,

    epochs=3,

    steps_per_epoch=files_train.shape[0] // batch_size,

    validation_data=valid_gen,

    validation_steps=files_valid.shape[0] // batch_size)

model.save_weights("weights.h5")
def colorize(orig, resized):

    predicted = model.predict(resized)[1]

    resized = np.concatenate((resized, predicted), axis=3)

    

    for pos in range(resized.shape[0]):

        resized[pos,:,:,0] = resized[pos,:,:,0]*100

        resized[pos,:,:,1:] = resized[pos,:,:,1:]*255 - 128

        resized[pos] = skimage.color.lab2rgb(resized[pos])

        

        temp = skimage.color.rgb2gray(orig[pos])

        skimage.io.imshow(orig[pos])

        plt.show()

        

        final_image = skimage.transform.resize(resized[pos], orig[pos].shape)

        skimage.io.imshow(final_image)

        plt.show()
files = files_train

orig = []

num = 5

resized = np.zeros((num, 256, 256, 1))

for pos in range(num):

    orig.append(skimage.io.imread(path + files[pos]))

    temp = skimage.transform.resize(orig[pos], (256, 256))

    if temp.shape == (256, 256, 3):

        temp = skimage.color.rgb2lab(temp)[:,:,0]

        temp = temp/100

    if temp.shape == (256, 256, 1):

        resized[pos] = temp

    else:

        resized[pos,:,:,0] = temp

colorize(orig, resized)
files = files_valid

orig = []

num = 5

resized = np.zeros((num, 256, 256, 1))

for pos in range(num):

    orig.append(skimage.io.imread(path + files[pos]))

    temp = skimage.transform.resize(orig[pos], (256, 256))

    if temp.shape == (256, 256, 3):

        temp = skimage.color.rgb2lab(temp)[:,:,0]

        temp = temp/100

    if temp.shape == (256, 256, 1):

        resized[pos] = temp

    else:

        resized[pos,:,:,0] = temp

colorize(orig, resized)