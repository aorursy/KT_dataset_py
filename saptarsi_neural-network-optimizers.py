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
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
modeladm = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
modelrms = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
modeladgrad = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
modelsgd = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
modelsgdm = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])
modeladm.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
modelrms.compile(optimizer='RMSprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
modeladgrad.compile(optimizer='adagrad',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
modelsgd.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
modelsgdm.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
hist4=modelsgd.fit(train_images, train_labels, epochs=10)
hist5=modelsgdm.fit(train_images, train_labels, epochs=10)
hist3=modeladgrad.fit(train_images, train_labels, epochs=10)
hist2=modelrms.fit(train_images, train_labels, epochs=10)
hist1=modeladm.fit(train_images, train_labels, epochs=10)
ep=np.arange(1,11,1)
acc1=hist1.history['accuracy']
acc2=hist2.history['accuracy']
acc3=hist3.history['accuracy']
acc4=hist4.history['accuracy']
acc5=hist5.history['accuracy']
list_of_tuples = list(zip(ep,acc1,acc2,acc3,acc4,acc5)) 
df = pd.DataFrame(list_of_tuples, columns = ['Epoch', 'Adam','RMS','Adagrad','SGD','SGDM']) 
df.index = df['Epoch']
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(.7, .95) # set the vertical range to [0-1]
plt.show()
modeladm.evaluate(test_images, test_labels)
modelrms.evaluate(test_images, test_labels)
modeladgrad.evaluate(test_images, test_label)
modelsgd.evaluate(test_images, test_labels)
modeladm.evaluate(test_images, test_labels)
hist1=modeladm.fit(train_images, train_labels, epochs=15)