# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import csv

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Any results you write to the current directory are saved as output.


import pandas as pd

import zipfile



sign_mnist_train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')



sign_mnist_test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')

sign_mnist_train
def get_data(filename):

    with open(filename) as training_file:

        csv_reader = csv.reader(training_file, delimiter=',')

        first_line = True

        temp_images = []

        temp_labels = []

        for row in csv_reader:

            if first_line:

                # print("Ignoring first line")

                first_line = False

            else:

                temp_labels.append(row[0])

                image_data = row[1:785]

                image_data_as_array = np.array_split(image_data, 28)

                temp_images.append(image_data_as_array)

        images = np.array(temp_images).astype('float')

        labels = np.array(temp_labels).astype('float')

    return images, labels





training_images, training_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

testing_images, testing_labels = get_data('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')



print(training_images.shape)

print(training_labels.shape)

print(testing_images.shape)

print(testing_labels.shape)
training_images = np.expand_dims(training_images, axis=3)

testing_images = np.expand_dims(testing_images, axis=3)



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode='nearest')



validation_datagen = ImageDataGenerator(

    rescale=1. / 255)



print(training_images.shape)

print(testing_images.shape)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])



model.compile(optimizer = tf.train.AdamOptimizer(),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=32),

                              steps_per_epoch=len(training_images) / 32,

                              epochs=15,

                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),

                              validation_steps=len(testing_images) / 32)



model.evaluate(testing_images, testing_labels)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()