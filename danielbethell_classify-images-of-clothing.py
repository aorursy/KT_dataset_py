from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import regularizers



# Helper libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from tensorflow.python.keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split



# Import Data

train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test= pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

print("Train size:{}\nTest size:{}".format(train.shape, test.shape))



# Transform Train and Test into images\labels.

x_train = train.drop(['label'], axis=1).values.astype('float32') # all pixel values

y_train = train['label'].values.astype('int32') # only labels i.e targets digits

x_test = test.drop(['label'], axis=1).values.astype('float32')

y_test = test['label'].values.astype('int32') # only labels i.e targets digits

x_train = x_train.reshape(x_train.shape[0], 28, 28) / 255.0

x_test = x_test.reshape(x_test.shape[0], 28, 28) / 255.0



print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.figure()

plt.imshow(x_train[0])

plt.colorbar()

plt.grid(False)

plt.show()
x_train = x_train / 255.0



x_test = x_test / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i], cmap=plt.cm.binary)

    plt.xlabel(class_names[y_train[i]])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)



print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
predictions[0]
np.argmax(predictions[0])
y_test[0]
def plot_image(i, predictions_array, true_label, img):

  predictions_array, true_label, img = predictions_array, true_label[i], img[i]

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

  predictions_array, true_label = predictions_array, true_label[i]

  plt.grid(False)

  plt.xticks(range(10))

  plt.yticks([])

  thisplot = plt.bar(range(10), predictions_array, color="#777777")

  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)



  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('blue')
i = 0

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions[i], y_test, x_test)

plt.subplot(1,2,2)

plot_value_array(i, predictions[i],  y_test)

plt.show()
i = 13

plt.figure(figsize=(6,3))

plt.subplot(1,2,1)

plot_image(i, predictions[i], y_test, x_test)

plt.subplot(1,2,2)

plot_value_array(i, predictions[i],  y_test)

plt.show()
num_rows = 5

num_cols = 3

num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):

  plt.subplot(num_rows, 2*num_cols, 2*i+1)

  plot_image(i, predictions[i], y_test, x_test)

  plt.subplot(num_rows, 2*num_cols, 2*i+2)

  plot_value_array(i, predictions[i], y_test)

plt.tight_layout()

plt.show()
img = x_test[1]



print(img.shape)
img = (np.expand_dims(img,0))



print(img.shape)
predictions_single = probability_model.predict(img)



print(predictions_single)
plot_value_array(1, predictions_single[0], y_test)

_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])