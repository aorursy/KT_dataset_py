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
# import

import tensorflow as tf

from tensorflow.keras import layers
train_raw = pd.read_csv('../input/train.csv')

test_raw = pd.read_csv('../input/test.csv')



train_label = train_raw['label']

train_data = train_raw.drop('label', axis = 1)
# normalize the data

train_data = train_data / 255

test_data = test_raw / 255



# seperate training examples for testing

held_data = train_data[-2000:]

held_label = train_label[-2000:]

train_data = train_data[:-2000]

train_label = train_label[:-2000]



# convert data from a data frame to numpy array and reshape

train_label = train_label.values

held_label = held_label.values

train_data = train_data.values.reshape([-1, 28, 28, 1])

test_data = test_data.values.reshape([-1, 28, 28, 1])

held_data = held_data.values.reshape([-1, 28, 28, 1])
model = tf.keras.models.Sequential()
model.add(

tf.keras.layers.Conv2D(

    filters = 40,

    kernel_size = [5,5],

    padding = 'same',

    activation = 'relu',

    input_shape = (28,28,1)

))



model.add(

tf.keras.layers.MaxPool2D(

    pool_size = [2,2],

    strides = 2

))



model.add(

tf.keras.layers.Conv2D(

    filters = 40,

    kernel_size = [5,5],

    padding = 'same',

    activation = 'relu',

    input_shape = (28,28,1)

))



model.add(

tf.keras.layers.MaxPool2D(

    pool_size = [2,2],

    strides = 2

))



model.add(

tf.keras.layers.Conv2D(

    filters = 80,

    kernel_size = [3,3],

    padding = 'same',

    activation = 'relu',

    input_shape = (28,28,1)

))



model.add(

tf.keras.layers.MaxPool2D(

    pool_size = [2,2],

    strides = 2

))



model.add(

tf.keras.layers.Dropout(0.10)

)
model.add(

tf.keras.layers.Flatten()

)



model.add(

tf.keras.layers.Dense(256, activation='relu')

)



model.add(

tf.keras.layers.Dropout(0.10)

)
model.add(

tf.keras.layers.Dense(10, activation='softmax')

)
model.compile(

    loss = 'sparse_categorical_crossentropy',

    optimizer = 'adam',

    metrics = ['accuracy']

)

history = model.fit(train_data, train_label, epochs=15)
import matplotlib.pyplot as plt



plt.plot(history.history['acc'])

plt.title('accuracy over epoch')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()
model.evaluate(held_data, held_label)
predict = model.predict_classes(test_data)

submit = pd.DataFrame(

    {"ImageId": list(range(1,len(predict)+1)),"Label": predict})



submit.to_csv("../mnist_output.csv",index=False)
print(os.listdir("../"))