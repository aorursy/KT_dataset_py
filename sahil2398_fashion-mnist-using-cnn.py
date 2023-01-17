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
import tensorflow as tf

from tensorflow import keras

import numpy as np
fashion_mnist = keras.datasets.fashion_mnist



(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0



test_images = test_images / 255.0
train_images = np.expand_dims(train_images, axis = -1)

train_labels = np.array(train_labels)
print(train_images.shape, train_labels.shape)
test_images = np.expand_dims(test_images, axis = -1)

test_labels = np.array(test_labels)
print(test_images.shape, test_labels.shape)
model = keras.Sequential([

    tf.keras.layers.Conv2D(16, (3,3),padding="same", activation = 'relu', input_shape = train_images.shape[1:]),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3),padding="same", activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3),padding="same", activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=13)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
import random

ri = random.randint(0,9999)
predictions[ri]
np.argmax(predictions[ri])
test_labels[ri]
import matplotlib.pyplot as plt
def plot_image(i, predictions_array, true_label, img):

  predictions_array, true_label, img = predictions_array, true_label[ri], img[ri]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])



  plt.imshow(img, cmap=plt.cm.binary)



  predicted_label = np.argmax(predictions_array)

  if predicted_label == true_label:

    color = 'blue'

  else:

    color = 'red'



  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],

                                100*np.max(predictions_array),

                                class_names[true_label]),

                                color=color)



def plot_value_array(i, predictions_array, true_label):

  predictions_array, true_label = predictions_array, true_label[ri]

  plt.grid(False)

  plt.xticks(range(10))

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)



  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
ri = random.randint(0,9999)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(ri, predictions[ri], test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(ri, predictions[ri],  test_labels)

plt.show()
ri = random.randint(0,9999)

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(ri, predictions[ri], test_labels, test_images)

plt.subplot(1,2,2)

plot_value_array(ri, predictions[ri],  test_labels)

plt.show()
# Correct predictions in blue and incorrect predictions in red.

num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for ri in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*ri+1)

  plot_image(ri, predictions[ri], test_labels, test_images)

  plt.subplot(num_rows, 2*num_cols, 2*ri+2)

  plot_value_array(ri, predictions[ri], test_labels)

plt.tight_layout()

plt.show()