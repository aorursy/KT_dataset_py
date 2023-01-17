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
# TensorFlow e tf.keras

import tensorflow as tf

from tensorflow import keras



# Librariesauxiliares

import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__)
import pandas as pd

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
print(train.shape, test.shape)
train_labels = train['label']
train_images = []

for i in range(0,len(train)):

    x = np.array(train.iloc[i][1:].values)

    x = x.reshape(28,28)

    train_images.append(x)

train_images = np.asarray(train_images) 
test_images = []

for i in range(0,len(test)):

    x = np.array(test.iloc[i].values)

    x = x.reshape(28,28)

    test_images.append(x)

test_images = np.asarray(test_images) 
plt.figure(figsize=(10,10))

for i in range(0,24):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(train_labels[i])

plt.show()



train_images = train_images / 255.0
test_images = test_images / 255.0
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap=plt.cm.binary)

    plt.xlabel(train_labels[i])

plt.show()
model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
###

train_images.shape, train_labels.shape
model.fit(train_images, train_labels, epochs=10)
predictions = model.predict(test_images)
np.argmax(predictions[0])
plt.figure(figsize=(10,10))

for i in range(36):

    plt.subplot(6,6,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_images[i], cmap=plt.cm.binary)

    plt.xlabel(np.argmax(predictions[i]))

plt.show()
img = test_images[4]

img.shape
img = (np.expand_dims(img,0))

img.shape
predictions_single = model.predict(img)

predictions_single
np.argmax(predictions_single[0])
print(type(predictions))

print(predictions.shape)

submission = pd.DataFrame()

for i in range(0,len(predictions)):

    submission = submission.append([

        {'ImageId': i, 'Label': np.argmax(predictions[i])}],

    ignore_index=True)

    
submission.shape
submission.head()
submission.to_csv('submission_italocosta_mnist.csv', index=False)