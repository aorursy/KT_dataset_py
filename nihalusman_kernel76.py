

import cv2

import gc

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import openslide

import os

import tensorflow as tf

from keras.applications import InceptionResNetV2

from keras import models

from keras import layers

from random import randint

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from tqdm import tqdm

%matplotlib inline



print(tf.__version__)

print(tf.keras.__version__)
import json

with open("/kaggle/input/iwildcam-2020-fgvc7/iwildcam2020_train_annotations.json") as file : 

    data = json.load(file)

    

train_table = pd.DataFrame(data['annotations'])

print(train_table.shape)

train_table.head()
index = np.random.rand(len(train_table)) < 0.85

train = train_table[index]

valid = train_table[~index]

print(train.shape)

print(valid.shape)

train.head()
train_paths = train["image_id"].apply(lambda x:  '/kaggle/input/iwildcam-2020-fgvc7/train/' + x + '.jpg').values

valid_paths = valid["image_id"].apply(lambda x:  '/kaggle/input/iwildcam-2020-fgvc7/train/' + x + '.jpg').values
train_labels = pd.get_dummies(train['category_id']).astype('int32').values

valid_labels = pd.get_dummies(valid['category_id']).astype('int32').values



print(train_labels.shape) 

print(valid_labels.shape)
BATCH_SIZE= 32 

img_size = 512

EPOCHS = 20

    
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels))

    .map(decode_image)

    .shuffle(512)

    .batch(BATCH_SIZE)

    )

valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((valid_paths, valid_labels))

    .map(decode_image)

    .batch(BATCH_SIZE)

)
conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(256,256,3))



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(30, activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])


history = model.fit(

            train_dataset, 

            validation_data = valid_dataset, 

            steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,            

            validation_steps=valid_labels.shape[0] // BATCH_SIZE,            

            epochs=EPOCHS,

)

model.save_weights('model_wieghts.h5')

model.save('model_keras.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



#Train and validation accuracy

plt.plot(epochs, acc, 'b', label='Training accurarcy')

plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')

plt.title('Training and Validation accurarcy')

plt.legend()



plt.figure()

#Train and validation loss

plt.plot(epochs, loss, 'b', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and Validation loss')

plt.legend()



plt.show()