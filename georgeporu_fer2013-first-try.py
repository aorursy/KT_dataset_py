# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense

from keras.layers import AveragePooling2D, BatchNormalization

from keras.layers import GlobalAveragePooling2D

from keras.models import Sequential

from keras.layers import Flatten

from keras.models import Model

from keras.layers import Input

from keras.layers import MaxPooling2D

from keras.layers import SeparableConv2D

from keras import layers

from keras.regularizers import l2



# metric to balance precision and recall

def fbeta(y_true, y_pred, threshold_shift=0):

    beta = 1



    # just in case of hipster activation at the final layer

    y_pred = K.clip(y_pred, 0, 1)



    # shifting the prediction threshold from .5 if needed

    y_pred_bin = K.round(y_pred + threshold_shift)



    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()

    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)

    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)



    precision = tp / (tp + fp)

    recall = tp / (tp + fn)



    beta_squared = beta ** 2

    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))



def preprocess_input(x):

    x = x.astype('float32')

    x = x / 255.0

    return x



def get_model2():

    model = Sequential()



    model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Convolution2D(128, (5, 5), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Convolution2D(512, (3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Convolution2D(512, (3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))

    model.add(Dropout(0.25))



    model.add(Flatten())



    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Dense(512))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.25))



    model.add(Dense(7))

    model.add(Activation('softmax'))



    return model



def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):

    regularization = l2(l2_regularization)



    # base

    img_input = Input(input_shape)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,

               use_bias=False)(img_input)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,

               use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)



    # module 1

    residual = Conv2D(16, (1, 1), strides=(2, 2),

                      padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(16, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(16, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    # module 2

    residual = Conv2D(32, (1, 1), strides=(2, 2),

                      padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(32, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(32, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    # module 3

    residual = Conv2D(64, (1, 1), strides=(2, 2),

                      padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(64, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(64, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    # module 4

    residual = Conv2D(128, (1, 1), strides=(2, 2),

                      padding='same', use_bias=False)(x)

    residual = BatchNormalization()(residual)



    x = SeparableConv2D(128, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)

    x = Activation('relu')(x)

    x = SeparableConv2D(128, (3, 3), padding='same',

                        kernel_regularizer=regularization,

                        use_bias=False)(x)

    x = BatchNormalization()(x)



    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])



    x = Conv2D(num_classes, (3, 3),

               # kernel_regularizer=regularization,

               padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    output = Activation('softmax', name='predictions')(x)



    model = Model(img_input, output)

    return model
import os

import sys

import cv2

import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

from math import sqrt

import scipy.io as io

import tensorflow as tf

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix



import random as rnd



from keras import backend as K

from keras.utils import np_utils

from keras.callbacks import CSVLogger

from keras.models import load_model

from keras.utils import np_utils as npu

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



# some hyperparameters

BATCH_SIZE = 64

NUM_EPOCHS = 10000



emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

num_classes = len(emotion_labels)



# loading the np arrays with image labels



# print(os.listdir("../input"))



data = np.load("../input/image-labels/Y_train.npz")

y_train = data['a']



data = np.load("../input/image-labels/Y_val.npz")

y_val = data['a']



data = np.load("../input/image-labels/Y_test.npz")

y_test = data['a']



# loading the np arrays with frontalized images

data = np.load("../input/new-train-raw/X_train_raw.npz")

X_train_raw = data['a']



data = np.load("../input/raw-images/X_val_raw.npz")

X_val_raw = data['a']



data = np.load("../input/raw-images/X_test_raw.npz")

X_test_raw = data['a']



num_train = X_train_raw.shape[0]

num_validation = X_val_raw.shape[0]

num_test = X_test_raw.shape[0]



print(num_train)

print(num_validation)

print(num_test)



# starting from the frontalized set

# boosted sets

X_train_boost_1 = X_train_raw.copy().astype(np.uint8)

y_train_boost_1 = y_train.copy()



X_train_boost_2 = X_train_raw.copy().astype(np.uint8)

y_train_boost_2 = y_train.copy()



X_train_boost_3 = X_train_raw.copy().astype(np.uint8)

y_train_boost_3 = y_train.copy()



X_train_boost_4 = X_train_raw.copy().astype(np.uint8)

y_train_boost_4 = y_train.copy()



X_train_boost_5 = X_train_raw.copy().astype(np.uint8)

y_train_boost_5 = y_train.copy()



X_train_boost_6 = X_train_raw.copy().astype(np.uint8)

y_train_boost_6 = y_train.copy()



X_train_boost_7 = X_train_raw.copy().astype(np.uint8)

y_train_boost_7 = y_train.copy()



X_train_boost_8 = X_train_raw.copy().astype(np.uint8)

y_train_boost_8 = y_train.copy()



X_train_boost_9 = X_train_raw.copy().astype(np.uint8)

y_train_boost_9 = y_train.copy()



X_train_boost_10 = X_train_raw.copy().astype(np.uint8)

y_train_boost_10 = y_train.copy()



for x in range(0, X_train_raw.shape[0]):

    idx = rnd.randint(0, X_train_raw.shape[0]-1)

    X_train_boost_1[x] = X_train_raw[idx]

    y_train_boost_1[x] = y_train[idx]



#loading models

bag_model_1 = mini_XCEPTION((48,48,1),7)

#bag_model_2 = mini_XCEPTION((48,48,1),7)

#bag_model_3 = mini_XCEPTION((48,48,1),7)

#bag_model_4 = mini_XCEPTION((48,48,1),7)

#bag_model_5 = mini_XCEPTION((48,48,1),7)

#bag_model_6 = mini_XCEPTION((48,48,1),7)

#bag_model_7 = mini_XCEPTION((48,48,1),7)

#bag_model_8 = mini_XCEPTION((48,48,1),7)



# augment and fit all datasets

datagen = ImageDataGenerator(

  featurewise_center=False,

  samplewise_center=False,

  featurewise_std_normalization=False,

  samplewise_std_normalization=False,

  zca_whitening=False,

  rotation_range=40,

  width_shift_range=0.2,

  height_shift_range=0.2,

  shear_range=0,

  zoom_range=0.1,

  horizontal_flip=True,

  vertical_flip=False)

datagen.fit(X_train_boost_1)

datagen.fit(X_val_raw)



# to be applied during training

reduce_lr = ReduceLROnPlateau(

  monitor='val_loss', factor=0.1,

  patience=int(50/4), verbose=0)

early_stop = EarlyStopping(

  monitor='val_loss', min_delta=0, patience=50,

  verbose=0, mode='auto')



# compiling models

bag_model_1.compile(

  loss='categorical_crossentropy',

  optimizer='adam',

  metrics=[fbeta, 'accuracy'])



# Training

train_flow_1 = datagen.flow(X_train_boost_1, y_train_boost_1, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_2 = datagen.flow(X_train_boost_2, y_train_boost_2, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_3 = datagen.flow(X_train_boost_3, y_train_boost_3, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_4 = datagen.flow(X_train_boost_4, y_train_boost_4, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_5 = datagen.flow(X_train_boost_5, y_train_boost_5, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_6 = datagen.flow(X_train_boost_6, y_train_boost_6, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_7 = datagen.flow(X_train_boost_7, y_train_boost_7, batch_size=BATCH_SIZE, shuffle=False)

#train_flow_8 = datagen.flow(X_train_boost_8, y_train_boost_8, batch_size=BATCH_SIZE, shuffle=False)



history1 = bag_model_1.fit_generator(

    train_flow_1,

    steps_per_epoch=num_train / BATCH_SIZE,

    epochs=NUM_EPOCHS,

    verbose=0,

    validation_data=(X_val_raw,y_val),

    validation_steps=num_validation / BATCH_SIZE,

    callbacks=[reduce_lr, early_stop])

bag_model_1.save('miniXCEPTION_raw.h5')



X_validation = np.vstack((X_val_raw,X_test_raw))

y_validation = np.vstack((y_val,y_test))

num_validation = num_validation*2



# evaluate

score = bag_model_1.evaluate(X_validation, y_validation, steps=int(num_validation / BATCH_SIZE))

print('miniXCEPTION raw Evaluation Loss: ', score[0])

print('miniXCEPTION raw Evaluation Accuracy: ', score[1])



# Print Model Stats

print('Training accuracy')

print(max(history1.history['acc']))

#print(max(history2.history['acc']))

#print(max(history3.history['acc']))

#print(max(history4.history['acc']))

#print(max(history5.history['acc']))

#print(max(history6.history['acc']))

#print(max(history7.history['acc']))

#print(max(history8.history['acc']))



print('Validation accuracy')

print(max(history1.history['val_acc']))

#print(max(history2.history['val_acc']))

#print(max(history3.history['val_acc']))

#print(max(history4.history['val_acc']))

#print(max(history5.history['val_acc']))

#print(max(history6.history['val_acc']))

#print(max(history7.history['val_acc']))

#print(max(history8.history['val_acc']))



plt.figure()

plt.plot(history1.history['acc'], color='C0', label='Training acc')

#plt.plot(history2.history['acc'], color='C1', label='M2 Training acc')

#plt.plot(history3.history['acc'], color='C2', label='M3 Training acc')

#plt.plot(history4.history['acc'], color='C3', label='M4 Training acc')

#plt.plot(history5.history['acc'], color='C4', label='M5 Training acc')

#plt.plot(history6.history['acc'], color='C5', label='M6 Training acc')

#plt.plot(history7.history['acc'], color='C6', label='M7 Training acc')

#plt.plot(history8.history['acc'], color='C7', label='M8 Training acc')

plt.title('one miniXCEPTION raw Training Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(loc='lower right')

plt.show()



plt.figure()

plt.plot(history1.history['loss'], color='C0', label='Training loss')

#plt.plot(history2.history['loss'], color='C1', label='M2 Training loss')

#plt.plot(history3.history['loss'], color='C2', label='M3 Training loss')

#plt.plot(history4.history['loss'], color='C3', label='M4 Training loss')

#plt.plot(history5.history['loss'], color='C4', label='M5 Training loss')

#plt.plot(history6.history['loss'], color='C5', label='M6 Training loss')

#plt.plot(history7.history['loss'], color='C6', label='M7 Training loss')

#plt.plot(history8.history['loss'], color='C7', label='M8 Training loss')

plt.title('one miniXCEPTION raw Training Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(loc='upper right')

plt.show()



y_pred = bag_model_1.predict(X_test_raw)

y_pred = np.argmax(y_pred,axis=1)

y_true = np.asarray([np.argmax(i) for i in y_test])

print(y_pred.shape)

print(y_true.shape)



cm = confusion_matrix(y_true, y_pred)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(

    cm_norm, annot=True, linewidths=0, square=False, cmap='Greens',

    yticklabels=emotion_labels, xticklabels=emotion_labels,

    vmin=0, vmax=np.max(cm_norm), fmt='.2f',

    annot_kws={'size': 20}

)

ax.set(xlabel='Predicted Label', ylabel='Actual Label', title='miniXCEPTION frontalized CM')

plt.show()
