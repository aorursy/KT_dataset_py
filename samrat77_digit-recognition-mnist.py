# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data from kaggle dataset

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv", header=0)

train_label = train_data['label']

train_img = train_data.iloc[:,1:] / 255  # scale

train_img = train_img.to_numpy()

train_img.resize(42000, 28, 28, 1)



test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv", header=0)

test_img = test_data.copy() / 255  # scale

test_img = test_img.to_numpy()

test_img.resize(28000, 28, 28, 1)



# check sample submission

submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv", header=0)

submission.head()
# plot some examples

samples = np.random.randint(0,12,3)

for i in samples:

    plt.imshow(np.resize(train_img[i], (28, 28)))

    plt.show()
# to get 1.00 accuracy, train on entire MNIST dataset, added from kaggle datasets in .csv

mnist_train = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")

mnist_test = pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")

mnist = pd.concat([mnist_train, mnist_test], axis=0)



# Get all mnist as training

mnist_train_label = mnist['label']

mnist_train_img = mnist.drop('label', axis=1).to_numpy()

mnist_train_img = mnist_train_img / 255  # scale

mnist_train_img.resize(70000, 28, 28, 1)
# build image data generator with keras

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    # Parameters for data augmentation:

     featurewise_center=True,

     featurewise_std_normalization=True,

     rotation_range=20,

     width_shift_range=0.2,

     height_shift_range=0.2,

     horizontal_flip=True 

)



datagen.fit(mnist_train_img)
# create the NN model



model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),

    tf.keras.layers.MaxPool2D(2,2),

#    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),

    tf.keras.layers.Conv2D(192, (3,3), activation='relu', padding='same'),

    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(192, (5,5), activation='relu', padding='same'),

    tf.keras.layers.MaxPool2D(2,2, padding='same'),

#    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=256, activation='relu'),

#    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(units=10, activation='softmax')

])



model.summary()



model.compile(loss='sparse_categorical_crossentropy',

              optimizer='adam',

              metrics=['acc'])
history = model.fit_generator(datagen.flow(mnist_train_img, mnist_train_label, batch_size=256),

                              epochs=50,

                              verbose=1,

                              validation_data=(train_img, train_label),

                              shuffle=True)
predictions = model.predict(test_img).argmax(axis=1)
results = pd.DataFrame({"ImageId": range(1, len(test_img)+1), "Label": predictions})

results.to_csv("predictions_submission.csv", header=True, index=False)