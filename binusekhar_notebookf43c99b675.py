import os

import numpy as np

import pandas as pd

import pydicom

from matplotlib import pyplot as plt

import cv2

from tensorflow.keras.applications.mobilenet import preprocess_input

import tensorflow as tensorflow

import gc

import csv

from datetime import datetime
files = []

path = '../input/rsna-pneumonia-detection-challenge/stage_2_train_images'

for dirpath,dirname,filenames in os.walk(r'../input/rsna-pneumonia-detection-challenge/stage_2_train_images'):

    for filename in filenames:

        files.append(filename)



print("Number of train images :",len(files))
from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.layers import Concatenate, UpSampling2D, Conv2D, Reshape

from tensorflow.keras.models import Model



image_size = 224

def create_model(trainable=True):

    model = MobileNet(input_shape=(image_size, image_size, 3), include_top=False, alpha=1.0, weights = "imagenet")



    for layer in model.layers:

        layer.trainable = trainable



    block1 = model.get_layer("conv_pw_1_relu").output

    block2 = model.get_layer("conv_pw_3_relu").output

    block3 = model.get_layer("conv_pw_5_relu").output

    block6 = model.get_layer("conv_pw_11_relu").output

    block7 = model.get_layer("conv_pw_13_relu").output



    x = Concatenate()([UpSampling2D()(block7), block6])

    x = Concatenate()([UpSampling2D()(x), block3])

    x = Concatenate()([UpSampling2D()(x), block2])

    x = Concatenate()([UpSampling2D()(x), block1])

    x =  UpSampling2D()(x)

    x = Conv2D(1, kernel_size=1, activation="sigmoid")(x)

    x = Reshape((image_size, image_size))(x)



    return Model(inputs=model.input, outputs=x)
model = create_model()

model.summary()
def dice_coefficient(y_true, y_pred):

    #### Add your code here ####

    numerator = 2 * tensorflow.reduce_sum(y_true * y_pred)

    denominator = tensorflow.reduce_sum(y_true + y_pred)



    return numerator / (denominator + tensorflow.keras.backend.epsilon()) #### Add your code here ####
import tensorflow as tf

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.backend import log, epsilon

def loss(y_true, y_pred):

    return binary_crossentropy(y_true, y_pred) - log(dice_coefficient(y_true, y_pred) + epsilon())



def iou_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1])

    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)

    return 1 - score



# mean iou as a metric

def mean_iou(y_true, y_pred):

    y_pred = tf.round(y_pred)    

    #intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])

    intersect = tf.reduce_sum(y_true * y_pred, axis=[1])

    #union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    union = tf.reduce_sum(y_true, axis=[1]) + tf.reduce_sum(y_pred, axis=[1])

    smooth = tf.ones(tf.shape(intersect))

    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import binary_crossentropy



optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#model.compile(loss=loss, optimizer=optimizer, metrics=[dice_coefficient])

model.compile(loss=iou_loss, optimizer=optimizer, metrics=[mean_iou])
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("model-{loss:.2f}.h5", monitor="loss", verbose=1, save_best_only=True,

                             save_weights_only=True, mode="min", period=1)

stop = EarlyStopping(monitor="loss", patience=5, mode="min")

reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.1, patience=3, min_lr=1e-6, verbose=1, mode="min")
mask_coord = {}

with open(os.path.join('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv'), mode='r') as infile:

    reader = csv.reader(infile)

    next(reader, None)

    for rows in reader:

        filename = rows[0]

        coord = rows[1:5]

        tgt = rows[5]

        if tgt == '1':

            coord = [int(float(i)) for i in coord]

            if filename in mask_coord:

                mask_coord[filename].append(coord)

            else:

                mask_coord[filename] = [coord]

print("Number of positive targets :",len(mask_coord))
size = len(files)

batch_size = 3000

batch_num = 1

#batch_num = int(size/batch_size + 1)

image_size = 224



for i in range (batch_num):

    x_train = np.zeros((batch_size, image_size, image_size,3))

    masks = np.zeros((batch_size,image_size,image_size))

    start = i * batch_size

    end = ((i+1) * batch_size)-1

    if (end > size):

        end = size

    print("Batch start and end points   : ",start,end,datetime.now())

    

    x=0

    for j in range(start,end):

        train_image = (pydicom.dcmread(path + '/' + files[j]))

        img = train_image.pixel_array

        img = cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

        img = np.stack((img,)*3, axis=-1)

        x_train[x] = preprocess_input(np.array(img, dtype=np.float32))

        

        filename = files[x].split('.')[0]

        if filename in mask_coord:

            mask = np.zeros((1024,1024))

            loc = mask_coord[filename]

            for i in range(len(loc)):

                X,Y,W,H = loc[i]

                mask[Y:Y+H, X:X+W] = 1

            mask = cv2.resize(mask, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC)

            masks[x]= mask

        x = x + 1

    

    print("Model training starts        : ",datetime.now())

    trail = model.fit(x_train, masks, epochs=20, batch_size=9, verbose=1, callbacks=[checkpoint, reduce_lr, stop])

    print("Model training ends          : ",datetime.now())

    

    del x_train

    del masks

    gc.collect()
from keras.utils.vis_utils import model_to_dot

from IPython.display import Image



def show_model(in_model):

    f = model_to_dot(in_model, show_shapes=True, rankdir='UD')

    return Image(f.create_png())

show_model(model)
plt.figure(figsize=(12,4))

plt.subplot(121)

plt.plot(trail.epoch, trail.history["loss"], label="Train IoU")

plt.legend()

plt.subplot(122)

plt.plot(trail.epoch, trail.history["mean_iou"], label="Train Mean IoU")

plt.legend()

plt.show()