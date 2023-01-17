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
train_data = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

test_data = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')
train_data.head()
trainY = train_data['label']

trainY = tf.one_hot(trainY, 10)

trainX = train_data.drop(labels=['label'], axis=1)

trainX = trainX/255.0

trainX = trainX.values.reshape(-1, 28, 28, 1)

testY = test_data['label']

testY = tf.one_hot(testY, 10)

testX = test_data.drop(labels=['label'], axis=1)

testX = testX/255.0

testX = testX.values.reshape(-1, 28, 28, 1)
model1 = keras.Sequential([

    

    keras.layers.Conv2D(36,kernel_size=5, activation='relu',input_shape=(28,28,1)),

    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(64,kernel_size=5,activation='relu'),

    keras.layers.MaxPool2D((2,2)),

    

    keras.layers.Flatten(),

    keras.layers.Dense(256, activation='relu'),

    keras.layers.Dropout(0.4),

    keras.layers.Dense(10),

    keras.layers.Softmax()

])



model1.compile(optimizer=keras.optimizers.Adam(),

             loss=tf.keras.losses.CategoricalCrossentropy(),

             metrics=['accuracy'])

history = model1.fit(trainX, trainY, batch_size=128, epochs=40, verbose=2)
test_loss, test_acc = model1.evaluate(testX, testY)

print(test_loss, test_acc)
model1.summary()
model1.save("MNIST_CNN.h5")
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

axes = plt.gca()

axes.set_ylim([0.95,1])

plt.show()