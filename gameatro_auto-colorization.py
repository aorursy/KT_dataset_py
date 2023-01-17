import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import os
PATH = "../input/auto-colorization-data/auto_colorization/Data"
def ImageLoader(type_data, batch_size):

    X_path = os.path.join(PATH, type_data, "X")

    Y_path = os.path.join(PATH, type_data, "Y")

    

    X_files = sorted(os.listdir(X_path))

    Y_files = sorted(os.listdir(Y_path))

    

    

    import random

    XY = list(zip(X_files, Y_files))

    random.shuffle(XY)

    X_files, Y_files = zip(*XY)

    

    del XY

    

    

    L = len(X_files)

    

    

    while True:

        

        batch_start = 0

        batch_end = batch_size

        

        while batch_start < L:

            end = min(batch_end, L)

            X = []

            Y = []

            for (x,y) in zip(X_files[batch_start:end],Y_files[batch_start:end]):

                X.append(np.reshape(np.load(os.path.join(X_path,x)),(224,224,3)))#.astype(np.float32))

                Y.append(np.reshape(np.load(os.path.join(Y_path,y)),(224,224,2)))#.astype(np.float32))

            X = np.array(X)

            Y = np.array(Y)

            

            yield(X,Y)

            

            batch_start += batch_size

            batch_end += batch_size

            
from keras.models import Sequential

from keras.layers import Conv2D, Dense, MaxPool2D
from keras.applications.vgg16 import VGG16

vgg = VGG16()
from keras.utils import plot_model
plot_model(vgg)
vgg.summary()
for i in range(len(vgg.layers)):

    vgg.layers[i].trainable = False
import tensorflow as tf
from keras.layers import Add, Conv2D, UpSampling2D, BatchNormalization, Input
batch_norm1 = BatchNormalization()(vgg.layers[13].input)

conv1 = Conv2D(256,(1,1),activation="relu",padding="same",name="reg_conv1")(batch_norm1)

upscale1 = UpSampling2D()(conv1)



batch_norm2 = BatchNormalization()(vgg.layers[9].input)

add1 = Add(name="Add1")([upscale1,batch_norm2])

conv2 = Conv2D(128,(3,3),activation="relu",padding="same",name="reg_conv2")(add1)

upscale2 = UpSampling2D()(conv2)



batch_norm3 = BatchNormalization()(vgg.layers[5].input)

add2 = Add(name="Add2")([upscale2,batch_norm3])

conv3 = Conv2D(64,(3,3),activation="relu",padding="same",name="reg_conv3")(add2)

upscale3 = UpSampling2D()(conv3)



batch_norm4 = BatchNormalization()(vgg.layers[2].input)

add3 = Add(name="Add3")([upscale3,batch_norm4])

conv4 = Conv2D(2,(3,3),activation="relu",padding="same",name="reg_conv4")(add3)
from keras import backend
def LReg(y_true, y_pred):

  return backend.sum([backend.square(y_true[:, :, 0] - y_pred[:, :, 0]), backend.square(y_true[:, :, 1] - y_pred[:, :, 1])],axis=0)
from keras.models import Model
color_model = Model(inputs=vgg.layers[0].output,outputs=conv4)

color_model.compile(optimizer="adam",loss="mse",metrics=[LReg])
color_model.summary()
plot_model(color_model)
batch_size = 30

epochs = 300

train_size = len(os.listdir(os.path.join(PATH, "Train","X")))

val_size = len(os.listdir(os.path.join(PATH, "Val","X")))
history = color_model.fit_generator(ImageLoader("Train", batch_size),steps_per_epoch=train_size//batch_size, epochs = epochs, validation_data = ImageLoader("Val",batch_size), validation_steps=val_size//batch_size)
color_model.save("color_model.h5")