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



# Any results you write to the current directory are saved as output.
!pip install tensorflow-datasets



# Import TensorFlow Datasets

import tensorflow as tf

import tensorflow_datasets as tfds



import logging

logger = tf.get_logger()

logger.setLevel(logging.ERROR)





# Helper libraries

import math

import numpy as np

import matplotlib.pyplot as plt



#enable eager execution for image.numpy() method

tf.enable_eager_execution()

tf.executing_eagerly()
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
num_train_examples = metadata.splits['train'].num_examples

num_test_examples = metadata.splits['test'].num_examples

print("Number of training examples: {}".format(num_train_examples))

print("Number of test examples:     {}".format(num_test_examples))
print(metadata.supervised_keys)



#Normalize function is getting Images, Label from the mnist metadata.supervised_keys

def normalize(images, labels):

  images = tf.cast(images, tf.float32)

  images /= 255

  return images, labels



# The map function applies the normalize function to each element in the train

# and test datasets. It is similar to df.apply()

train_dataset =  train_dataset.map(normalize)

test_dataset  =  test_dataset.map(normalize)



# The first time you use the dataset, the images will be loaded from disk

# Caching will keep them in memory, making training faster

train_dataset =  train_dataset.cache()

test_dataset  =  test_dataset.cache()
tf.executing_eagerly()
# Take a single image, and remove the color dimension by reshaping

for image, label in test_dataset.take(1):

  break

image = image.numpy().reshape((28,28))



# Plot the image - voila a piece of fashion clothing

plt.figure()

plt.imshow(image, cmap=plt.cm.binary)



#Note the reason why we created class_names[] is because the label is a number in Dataset

plt.xlabel(label)

plt.colorbar()

plt.grid(False)

plt.show()
model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

    tf.keras.layers.Dense(128, activation=tf.nn.relu),

    tf.keras.layers.Dense(10,  activation=tf.nn.softmax)

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
BATCH_SIZE = 32

model_1 = model.fit(train_dataset.batch(BATCH_SIZE), epochs=5)
model.save('Fashion_Classifier_Model.h5') 