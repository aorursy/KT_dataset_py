import numpy as np

import keras

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten

from keras.models import Model

from keras import optimizers

from keras.utils.np_utils import to_categorical

from keras.layers import concatenate,AveragePooling2D

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau

from keras.regularizers import l2

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input, decode_predictions

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.layers import Input

import os

import cv2

import scipy

import skimage

from keras.callbacks import ModelCheckpoint

from keras.utils.vis_utils import plot_model

import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

import random



train_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TRAIN/'

test_path = '../input/blood-cells/dataset2-master/dataset2-master/images/TEST/'



classes = []

for cell in os.listdir(train_path):

    classes.append(cell)

class_to_ind = dict(zip(classes, range(len(classes))))

ind_to_class = dict(zip(range(len(classes)), classes))

    

def load_sample_imgs(path):

    rows = 10

    cols = 5

    sorted_dirs = sorted(os.listdir(path))

    fig, axes = plt.subplots(rows,cols, figsize=(30,10))

    class_arr = ['MONOCYTE', "EOSINOPHIL", 'NEUTROPHIL','LYMPHOCYTE' ]

    for i in range(rows):

        for j in range(cols):

            cell = random.choice(class_arr)

            all_files = os.listdir(path + '/' + cell)

            rand_img = random.choice(all_files)

            img = plt.imread(path + '/' + cell + '/' + rand_img)

            axes[i][j].imshow(img)

            ec = (0, .6, .1)

            fc = (0, .7, .2)

            axes[i][j].text(0, -20, cell, size=10, rotation=0,

                ha="left", va="top", 

                bbox=dict(boxstyle="round", ec=ec, fc=fc))

    plt.setp(axes, xticks=[], yticks=[])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    

load_sample_imgs(train_path)





    

print(class_to_ind)

print(ind_to_class)
def get_data(path):

    X = []

    y = []

    for cell in class_to_ind:

        for image_name in os.listdir(path + '/' + cell):

            img_file = cv2.imread(path + '/' + cell + '/' + image_name)

            if img_file is not None: 

                img_file = cv2.resize(img_file, (60,80))

                img = np.asarray(img_file)

                X.append(img)

                y.append(class_to_ind[cell])

    X = np.asarray(X)

    y = np.asarray(y)

    return X, y

                

X_train, y_train = get_data(train_path)

X_test, y_test = get_data(test_path)

print('done loading data')

print(X_train.shape)
y_train_cat = to_categorical(y_train, num_classes=4)

y_test_cat = to_categorical(y_test, num_classes=4)



print(y_train_cat.shape)

print(y_test_cat.shape)
train_datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False, # randomly flip images"

    zoom_range=[.8, 1],

    channel_shift_range=30,

    fill_mode='reflect')



train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=32)



test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=32)
inp = Input(shape=(80,60,3))

k = BatchNormalization()(inp)

k = Conv2D(32, (7,7), padding="same",activation="relu",strides=(2,2))(k)

k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k) 

k = Conv2D(32, (3,3), padding="same",activation="relu",strides=(1,1))(k)

k = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(k)

for j in range(1,4+1):

    out_conv = []

    for i in [(1,1),(3,3),(5,5),(0,0)]:

        p = k

        if i == (1,1):

            p = Conv2D(32, (1,1), padding="same",activation="relu")(p)

            out_conv.append(Conv2D(32, (1,1), padding="same",activation="relu")(p))

        elif i == (0,0):

            p = MaxPool2D(pool_size=(2, 2), padding="same",strides=(1,1))(p)

            out_conv.append(Conv2D(32, (1,1), padding="same",activation="relu")(p))

        else:

            p = Conv2D(32, (1,1), padding="same",activation="relu")(p)

            p = Conv2D(32, i, padding="same",activation="relu")(p)

            out_conv.append(Conv2D(32, i, padding="same",activation="relu")(p))

    x = concatenate(out_conv, axis = -1)

    #if j%2 == 0:

    #    x = MaxPool2D(pool_size=(3, 3), padding="same",strides=(2,2))(x)

    #x = BatchNormalization(axis=-1)(x)

    k = x

# x = Dropout(0.5)(k)

x = MaxPool2D(pool_size=(7, 7), padding="same",strides=(2,2))(x)

x = Flatten()(x)

#x = Dense(1024,activation="relu")(x)

#x = BatchNormalization()(x)

#x = Dropout(0.5)(x)

y = Dense(4,activation="softmax")(x)

#    z = Dense(2,activation="softmax")(x)

model = Model(inp, y)

opt = optimizers.Adam(lr=0.01,decay=0.0001)

model.compile(loss='categorical_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])

print(model.summary())
history = model.fit(X_train,

                    y_train_cat,

                    batch_size = 32,

                    epochs=5,

                    validation_data = (X_test, y_test_cat))
y_test_pred = model.evaluate(X_test, y_test_cat, verbose=1)
print(y_test_pred)
for cell in class_to_ind:

    for i in range(5):

        image_arr = os.listdir(test_path + '/' + cell)

        random_img = random.choice(image_arr)

        img_file = cv2.imread(test_path + '/' + cell + '/' + random_img)

        if img_file is not None:

            img_file = cv2.resize(img_file, (60,80))

            img = np.asarray(img_file)

            X = []

            X.append(img)

            X = np.asarray(X)

            cell_pred = model.predict(X)

            cell_top_pred = np.argmax(cell_pred, axis=1)

            print('Current cell: ' + cell)

            print('Current Prediction: ' + ind_to_class[cell_top_pred[0]])

            