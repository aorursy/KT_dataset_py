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
# Import libraries

from __future__ import absolute_import, division, print_function, unicode_literals





# Import TensorFlow

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)



# Helper libraries

import math

import numpy as np

import matplotlib.pyplot as plt



# Improve progress bar display

import tqdm

import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm





print(tf.__version__)



# This will go away in the future.

# If this gives an error, you might be running TensorFlow 2 or above

# If so, then just comment out this line and run this cell again

tf.enable_eager_execution()  
train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")
# Split the tradatasets into features and labels



features_train = train_dataset.iloc[:, 1:]

labels_train = train_dataset.iloc[:, 0:1].values
training_dataset = (

    tf.data.Dataset.from_tensor_slices(

        (

            tf.cast(features_train.values, tf.float32),

            tf.cast(labels_train, tf.int32)

        )

    )

)
testing_dataset = (

    tf.data.Dataset.from_tensor_slices(

        (

            tf.cast(test_dataset.values, tf.float32)

        )

    )

)
# The map function applies the normalize function to each element in the train

# and test datasets

def normalize(images, labels):

  images = tf.cast(images, tf.float32)

  images /= 255

  return images, labels



# The map function applies the normalize function to each element in the train

# and test datasets

training_dataset = training_dataset.map(normalize)



def reshape_it(images, labels):

    images = tf.reshape(images, [28,28,1])

    return images, labels



training_dataset = training_dataset.map(reshape_it)
print(training_dataset)
# Take a single image in the training dataset, reshape and plot it

for image, label in training_dataset.take(1):

  break

image = image.numpy().reshape((28,28))



# Plot the image

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)

plt.colorbar()

plt.grid(False)

plt.show()
# Verify the data in the training dataset

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

plt.figure(figsize=(10,10))

i = 0

for (image, label) in training_dataset.take(25):

    image = image.numpy().reshape((28,28))

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(image, cmap=plt.cm.binary)

    plt.xlabel(class_names[label])

    i += 1

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

    tf.keras.layers.MaxPooling2D((2, 2), strides=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

])
model.compile(optimizer='adam', 

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
print(training_dataset)
BATCH_SIZE = 32

TRAIN_DATA_SIZE = len(train_dataset.index)



training_dataset = training_dataset.repeat().shuffle(TRAIN_DATA_SIZE).batch(BATCH_SIZE)
print(training_dataset)
model.fit(training_dataset, epochs=10, steps_per_epoch=math.ceil(TRAIN_DATA_SIZE/BATCH_SIZE))
def reshape_test(images):

    images = tf.reshape(images, [-1, 28,28,1])

    return images



testing_dataset = testing_dataset.map(reshape_test)
print(testing_dataset)
# Generate the predictions

TEST_DATA_SIZE = len(test_dataset.index)



output = model.predict(testing_dataset,steps=math.ceil(TEST_DATA_SIZE), verbose=True)
# Print the prediction for the first digit by getting the index of the highest associated probability

print(np.argmax(output[0]))
# Create the output CSV

import csv



with open('predictions.csv', 'w') as outfile:

    writer = csv.writer(outfile)

    writer.writerow(["ImageId", "Label"])

    for itemid, prediction in enumerate(output, start=1):

        # Our labels match the index, so we can just write the argmax instead of looking it up in

        # the class names

        writer.writerow([itemid, np.argmax(prediction)])