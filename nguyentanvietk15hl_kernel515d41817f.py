# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__)
fashion_mnist=keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
classes = np.unique(train_labels)

num_classes = len(classes)

num_classes
print(train_images[0])
import PIL
from PIL import Image

train_imgres=[]

test_imgres=[]

for i in range(train_images.shape[0]):

    new_img_arr = np.array(Image.fromarray(train_images[i]).resize((32, 32), Image.ANTIALIAS))

    train_imgres.append(new_img_arr)

for i in range(test_images.shape[0]):

    new_img_arr = np.array(Image.fromarray(test_images[i]).resize((32, 32), Image.ANTIALIAS))

    test_imgres.append(new_img_arr)


train_imgres=np.asarray(train_imgres)

train_imgres.shape
X_train=[]

X_test=[]

for i in range(train_images.shape[0]):

      X_train.append(np.dstack([train_imgres[i]] * 3))

for i in range(test_images.shape[0]):

      X_test.append(np.dstack([test_imgres[i]] * 3))
X_train=np.asarray(X_train)

print(X_train.shape)

X_test=np.asarray(X_test)

print(X_test.shape)
print(X_train[0])
X_train = X_train / 255.

X_test= X_test / 255.

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')
X_train[0]
from keras.utils import to_categorical

from keras.applications.vgg19 import preprocess_input

from sklearn.model_selection import train_test_split

train_Y_one_hot = to_categorical(train_labels)

test_Y_one_hot = to_categorical(test_labels)
train_X,valid_X,train_label,valid_label = train_test_split(X_train,

                                                           train_Y_one_hot,

                                                           test_size=0.2,

                                                           random_state=13

                                                           )
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

IMG_WIDTH = 32

IMG_HEIGHT = 32

IMG_DEPTH = 3

BATCH_SIZE = 16
train_X = preprocess_input(train_X)

valid_X = preprocess_input(valid_X)

test_X  = preprocess_input (X_test)
plt.figure()

plt.imshow(X_train[5])

plt.colorbar()

plt.grid(False)

plt.show()
class_names = ['Áo thun', 'Quần dài', 'Áo liền quần', 'Đầm', 'Áo khoác',

               'Sandal', 'Áo sơ mi', 'Giày', 'Túi xách', 'Ủng']
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[train_labels[i]])

plt.show()
from keras.applications import VGG19

model_vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))
train_features = model_vgg19.predict(np.array(train_X), batch_size=BATCH_SIZE, verbose=1)

test_features = model_vgg19.predict(np.array(test_X), batch_size=BATCH_SIZE, verbose=1)

val_features = model_vgg19.predict(np.array(valid_X), batch_size=BATCH_SIZE, verbose=1)
print(train_features.shape, "\n",  test_features.shape, "\n", val_features.shape)

print(train_features.shape, "\n",  test_features.shape, "\n", val_features.shape)

train_features_flat = np.reshape(train_features, (48000, 1*1*512))

test_features_flat = np.reshape(test_features, (10000, 1*1*512))

val_features_flat = np.reshape(val_features, (12000, 1*1*512))
from keras import models

from keras.models import Model

from keras import layers

from keras import optimizers

from keras import callbacks

from keras.layers.advanced_activations import LeakyReLU
NB_TRAIN_SAMPLES = train_features_flat.shape[0]

NB_VALIDATION_SAMPLES = val_features_flat.shape[0]

NB_EPOCHS = 100



model = models.Sequential()

model.add(layers.Dense(512, activation='relu', input_dim=(1*1*512)))

model.add(layers.LeakyReLU(alpha=0.1))

model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(

    loss='categorical_crossentropy',

    optimizer=optimizers.Adam(),

  # optimizer=optimizers.RMSprop(lr=2e-5),

    metrics=['acc'])
model.summary()
reduce_learning = callbacks.ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.2,

    patience=2,

    verbose=1,

    mode='auto',

    epsilon=0.0001,

    cooldown=2,

    min_lr=0)



eary_stopping = callbacks.EarlyStopping(

    monitor='val_loss',

    min_delta=0,

    patience=7,

    verbose=1,

    mode='auto')



callbacks = [reduce_learning, eary_stopping]
history = model.fit(

    train_features_flat,

    train_label,

    epochs=NB_EPOCHS,

    validation_data=(val_features_flat, valid_label),

    callbacks=callbacks

)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)



plt.title('Training and validation accuracy')

plt.plot(epochs, acc, 'red', label='Training acc')

plt.plot(epochs, val_acc, 'blue', label='Validation acc')

plt.legend()



plt.figure()

plt.title('Training and validation loss')

plt.plot(epochs, loss, 'red', label='Training loss')

plt.plot(epochs, val_loss, 'blue', label='Validation loss')



plt.legend()



plt.show()
test_X[0].shape
plt.figure()

plt.imshow(test_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
predictions_single = model.predict(test_features[0])

print(np.argmax(predictions_single))
result=[]

for i in range(test_features.shape[0]):

     predictions_single = model.predict(test_features[i])

     result.append(np.argmax(predictions_single))

print(result)