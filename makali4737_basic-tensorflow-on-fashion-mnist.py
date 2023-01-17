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





from tensorflow.keras import datasets, layers, models

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

# from keras.models import Sequential

# from keras.layers import Dense

# from keras.layers import Dropout

# from keras.layers import Flatten

# from keras.constraints import maxnorm

# from keras.optimizers import SGD

# from keras.layers.convolutional import Conv2D

# from keras.layers.convolutional import MaxPooling2D

# from keras.utils import np_utils



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pathlib



import numpy

import scipy.ndimage as ndimage

import os

print(tf.__version__)
dataset = tf.data.Dataset.from_tensor_slices([8,3,0,8,2,1])

print(dataset)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
len(train_labels)
test_images.shape
len(test_labels)
plt.figure()

plt.imshow(train_images[0])

plt.colorbar()

plt.grid(False)

plt.show()
train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

  plt.subplot(5,5,i+1)

  plt.xticks([])

  plt.yticks([])

  plt.grid(False)

  plt.imshow(train_images[i], cmap = plt.cm.binary)

  plt.xlabel(class_names[train_labels[i]])

plt.show()
model =  keras.Sequential([

  keras.layers.Flatten(input_shape=(28, 28)),

  keras.layers.Dense(256, activation='relu'),

  keras.layers.Dense(128, activation='relu'),

  keras.layers.Dense(10, activation='softmax')

])



# model = models.Sequential()

# model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# model.add(layers.MaxPooling2D((2, 2), strides=2))

# model.add(layers.Conv2D(4, (3, 3), activation='relu'))

# model.add(layers.MaxPooling2D((2, 2), strides=2))

# model.add(layers.Flatten())

# model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',

             loss='sparse_categorical_crossentropy',

             metrics=['accuracy'])
history = model.fit(train_images,train_labels,epochs=12)

#model.fit(train_images,train_labels,epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\n Test accuracy:', test_acc)
accuracy = history.history['accuracy']

loss = history.history['loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.title('Training accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.title('Training loss')

plt.legend()

plt.show()
predictions = model.predict(test_images)

predictions[0]
np.argmax(predictions[0]) # Predicted value
test_labels[0] # Real Value
img=test_images[0]

img = (np.expand_dims(img,0))

predictions_single = model.predict(img)
def plot_value_array(i,predictions_array,true_label):

  predictions_array,true_label = predictions_array[i], true_label[i]

  plt.grid(False)

  plt.xticks([])

  plt.yticks([])

  thisplot = plt.bar(range(10),predictions_array,color='red')

  plt.ylim([0,1])

  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')

  thisplot[true_label].set_color('grey')
plot_value_array(0,predictions_single, test_labels)

_= plt.xticks(range(10),class_names,rotation=45)
np.argmax(predictions_single[0])