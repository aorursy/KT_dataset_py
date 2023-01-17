# to prevent unnecessary warnings

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# TensorFlow and tf.keras

import tensorflow as tf



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt

import os

import subprocess

import cv2

import json

import requests

from tqdm import tqdm



print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



print('\nTrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))

print('Test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
# reshape for feeding into the model

train_images_gr = train_images.reshape(train_images.shape[0], 28, 28, 1)

test_images_gr = test_images.reshape(test_images.shape[0], 28, 28, 1)



print('\nTrain_images.shape: {}, of {}'.format(train_images_gr.shape, train_images_gr.dtype))

print('Test_images.shape: {}, of {}'.format(test_images_gr.shape, test_images_gr.dtype))
fig, ax = plt.subplots(2, 5, figsize=(12, 6))

c = 0

for i in range(10):

    idx = i // 5

    idy = i % 5 

    ax[idx, idy].imshow(train_images_gr[i].reshape(28,28))

    ax[idx, idy].set_title(class_names[train_labels[i]])
# define input shape

INPUT_SHAPE = (28, 28, 1)



# define sequential model

model = tf.keras.models.Sequential()

# define conv-pool layers - set 1

model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), 

                                activation='relu', padding='valid', input_shape=INPUT_SHAPE))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

# define conv-pool layers - set 2

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), 

                                activation='relu', padding='valid', input_shape=INPUT_SHAPE))

model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))



# add flatten layer

model.add(tf.keras.layers.Flatten())



# add dense layers with some dropout

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.3))

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.3))



# add output layer

model.add(tf.keras.layers.Dense(10, activation='softmax'))



# compile model

model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])



# view model layers

model.summary()
EPOCHS = 20

train_images_scaled = train_images_gr / 255.

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, 

                                               restore_best_weights=True,

                                               verbose=1)



history = model.fit(train_images_scaled, train_labels,

                    batch_size=32,

                    callbacks=[es_callback], 

                    validation_split=0.1, epochs=EPOCHS,

                    verbose=1)
import pandas as pd



fig, ax = plt.subplots(1, 2, figsize=(10, 4))



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(kind='line', ax=ax[0])
import pandas as pd



fig, ax = plt.subplots(1, 2, figsize=(10, 4))



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(kind='line', ax=ax[0])

history_df[['accuracy', 'val_accuracy']].plot(kind='line', ax=ax[1]);
test_images_scaled = test_images_gr / 255.

predictions = model.predict(test_images_scaled)

predictions[:5]
prediction_labels = np.argmax(predictions, axis=1)

prediction_labels[:5]
from sklearn.metrics import confusion_matrix, classification_report

import pandas as pd



print(classification_report(test_labels, prediction_labels, target_names=class_names))

pd.DataFrame(confusion_matrix(test_labels, prediction_labels), index=class_names, columns=class_names)
print(test_labels[:100])
test_labels
test_image_idxs = [0, 23, 28]

test_labels[test_image_idxs]
layer_outputs = [layer.output for layer in model.layers]

activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
layer_outputs 

activation_model 
f, axarr = plt.subplots(3,4, figsize=(8, 5))



FIRST_IMAGE=0

SECOND_IMAGE=23

THIRD_IMAGE=28

CONVOLUTION_NUMBER = 13



for x in range(0,4):

  f1 = activation_model.predict(test_images_scaled[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[0,x].grid(False)

  axarr[0,x].set_title(class_names[test_labels[FIRST_IMAGE]])

  f2 = activation_model.predict(test_images_scaled[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[1,x].grid(False)

  axarr[1,x].set_title(class_names[test_labels[SECOND_IMAGE]])

  f3 = activation_model.predict(test_images_scaled[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[2,x].grid(False)

  axarr[2,x].set_title(class_names[test_labels[THIRD_IMAGE]])

plt.tight_layout()
f, axarr = plt.subplots(3,4, figsize=(8, 5))



FIRST_IMAGE=2

SECOND_IMAGE=3

THIRD_IMAGE=5

CONVOLUTION_NUMBER = 13



for x in range(0,4):

  f1 = activation_model.predict(test_images_scaled[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[0,x].grid(False)

  axarr[0,x].set_title(class_names[test_labels[FIRST_IMAGE]])

  f2 = activation_model.predict(test_images_scaled[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[1,x].grid(False)

  axarr[1,x].set_title(class_names[test_labels[SECOND_IMAGE]])

  f3 = activation_model.predict(test_images_scaled[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[2,x].grid(False)

  axarr[2,x].set_title(class_names[test_labels[THIRD_IMAGE]])

plt.tight_layout()
f, axarr = plt.subplots(3,4, figsize=(8, 5))



FIRST_IMAGE=2

SECOND_IMAGE=3

THIRD_IMAGE=5

CONVOLUTION_NUMBER = 3



for x in range(0,4):

  f1 = activation_model.predict(test_images_scaled[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[0,x].grid(False)

  axarr[0,x].set_title(class_names[test_labels[FIRST_IMAGE]])

  f2 = activation_model.predict(test_images_scaled[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[1,x].grid(False)

  axarr[1,x].set_title(class_names[test_labels[SECOND_IMAGE]])

  f3 = activation_model.predict(test_images_scaled[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]

  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='binary_r')

  axarr[2,x].grid(False)

  axarr[2,x].set_title(class_names[test_labels[THIRD_IMAGE]])

plt.tight_layout()
train_images_3ch = np.stack([train_images]*3, axis=-1)

test_images_3ch = np.stack([test_images]*3, axis=-1)



print('\nTrain_images.shape: {}, of {}'.format(train_images_3ch.shape, train_images_3ch.dtype))

print('Test_images.shape: {}, of {}'.format(test_images_3ch.shape, test_images_3ch.dtype))
import cv2



def resize_image_array(img, img_size_dims):

    img = cv2.resize(img, dsize=img_size_dims, 

                     interpolation=cv2.INTER_CUBIC)

    img = np.array(img, dtype=np.float32)

    return img
%%time



IMG_DIMS = (32, 32)



train_images_3ch = np.array([resize_image_array(img, img_size_dims=IMG_DIMS) for img in train_images_3ch])

test_images_3ch = np.array([resize_image_array(img, img_size_dims=IMG_DIMS) for img in test_images_3ch])



print('\nTrain_images.shape: {}, of {}'.format(train_images_3ch.shape, train_images_3ch.dtype))

print('Test_images.shape: {}, of {}'.format(test_images_3ch.shape, test_images_3ch.dtype))
# define input shape

INPUT_SHAPE = (32, 32, 3)



# get the VGG19 model

vgg_layers = tf.keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, 

                                               input_shape=INPUT_SHAPE)



vgg_layers.summary()
# Fine-tune all the layers

for layer in vgg_layers.layers:

    layer.trainable = True



# Check the trainable status of the individual layers

for layer in vgg_layers.layers:

    print(layer, layer.trainable)
# define sequential model

model = tf.keras.models.Sequential()



# Add the vgg convolutional base model

model.add(vgg_layers)



# add flatten layer

model.add(tf.keras.layers.Flatten())



# add dense layers with some dropout

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.3))

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dropout(rate=0.3))



# add output layer

model.add(tf.keras.layers.Dense(10, activation='softmax'))



# compile model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), 

              loss='sparse_categorical_crossentropy', 

              metrics=['accuracy'])



# view model layers

model.summary()
EPOCHS = 20

train_images_3ch_scaled = train_images_3ch / 255.

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, 

                                               restore_best_weights=True,

                                               verbose=1)



history = model.fit(train_images_3ch_scaled, train_labels,

                    batch_size=32,

                    callbacks=[es_callback], 

                    validation_split=0.1, epochs=EPOCHS,

                    verbose=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(kind='line', ax=ax[0])

history_df[['accuracy', 'val_accuracy']].plot(kind='line', ax=ax[1]);
test_images_3ch_scaled = test_images_3ch / 255.

predictions = model.predict(test_images_3ch_scaled)

predictions[:5]
prediction_labels = np.argmax(predictions, axis=1)

prediction_labels[:5]
print(classification_report(test_labels, prediction_labels, target_names=class_names))

pd.DataFrame(confusion_matrix(test_labels, prediction_labels), index=class_names, columns=class_names)