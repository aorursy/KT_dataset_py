import os

import sys

import random



import numpy as np

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

import cv2

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras
train_data_path = "../input/stage1_train/"

img_size = 128

train_ids = next(os.walk(train_data_path))[1]



x_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.uint8)

y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.bool)



for i, id_ in enumerate(train_ids):

    path = train_data_path+"{}/images/{}".format(id_, id_)

    img = cv2.imread(path+".png", 1)

    img = cv2.resize(img, (img_size, img_size))

    x_train[i]=img

    

    height, width, _ = img.shape

    label = np.zeros((height, width, 1))

    path2 = train_data_path+"{}/masks/".format(id_)

    for mask_file in next(os.walk(path2))[2]:

        mask_ = cv2.imread(path2+mask_file, 0)

        mask_ = cv2.resize(mask_, (img_size, img_size))

        mask_ = np.expand_dims(mask_, axis=-1)

        label = np.maximum(label, mask_)

        y_train[i]=label
ymg = x_train[1]

imshow(ymg)

plt.show()
test_image = y_train[1][:,:,0]

imshow(test_image)

plt.show()
fig = plt.figure(figsize=(12,12))

rows = 4

columns = 2

c = 0

counter = 0



for i in range(1, (rows*columns)+1):

    fig.add_subplot(rows, columns, i)

    if counter%2 == 0:

        plt.imshow(x_train[c][:,:,0])

    else:

        plt.imshow(y_train[c][:,:,0])

        c = c+1

    counter = counter+1

        
print("Original image shape : {}".format(x_train.shape))
from skimage.color import rgb2gray



x_train_ = rgb2gray(x_train)

print(x_train_.shape)
x_train = np.expand_dims(x_train_, axis=-1)

print(x_train.shape)
ymg = x_train[1][:,:,0]

imshow(ymg)

plt.show()
from keras.preprocessing.image import ImageDataGenerator



train_gen = ImageDataGenerator(rescale = 1./255,

                              rotation_range=40,

                              height_shift_range = 0.2,

                              width_shift_range = 0.2,

                              zoom_range = 0.2,

                              horizontal_flip = True,

                              fill_mode='nearest')



val_gen = ImageDataGenerator(rescale=1./255)
from sklearn.model_selection import train_test_split



xt, xv, yt, yv = train_test_split(x_train, y_train, test_size=0.2)
args = dict(batch_size = 32,

            shuffle=True)



def combineGenerator(gen1, gen2):

    while True:

        yield(gen1.next(), gen2.next())



train_generator_image = train_gen.flow(xt,**args)

train_generator_mask = train_gen.flow(yt, **args)

train_generator = combineGenerator(train_generator_image, train_generator_mask)

validation_generator_image = val_gen.flow(xv, **args)

validation_generator_mask = val_gen.flow(yv, **args)

validation_generator = combineGenerator(validation_generator_image, validation_generator_mask)
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    p = keras.layers.MaxPool2D(2, 2)(c)

    return c, p

    

    return c,p



def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):

    us = keras.layers.UpSampling2D((2,2))(x)

    concat = keras.layers.Concatenate()([us, skip])

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(concat)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c) 

    return c



def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)

    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)

    return c

    
def Unet():

    f = [16,32,64,128,256]

    img_=128

    inputs = keras.layers.Input((img_, img_, 1))

        

    p0 = inputs

    c1, p1 = down_block(p0, f[0]) #128 -> 64

    c2, p2 = down_block(p1, f[1]) #64 -> 32

    c3, p3 = down_block(p2, f[2]) #32 -> 16

    c4, p4 = down_block(p3, f[3]) #16->8

    

    bn = bottleneck(p4, f[4])

    

    u1 = up_block(bn, c4, f[3]) #8 -> 16

    u2 = up_block(u1, c3, f[2]) #16 -> 32

    u3 = up_block(u2, c2, f[1]) #32 -> 64

    u4 = up_block(u3, c1, f[0]) #64 -> 128

    

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)

    model = keras.models.Model(inputs, outputs)

    return model

    
model = Unet()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

model.summary()
train_steps = len(xt)//32

val_steps = len(xv)//32

history = model.fit_generator(train_generator, steps_per_epoch = train_steps, epochs = 15, validation_data = validation_generator, validation_steps = val_steps)