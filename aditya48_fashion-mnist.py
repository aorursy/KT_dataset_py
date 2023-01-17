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
import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt
fashion = tf.keras.datasets.fashion_mnist

(train_x,train_y),(test_x,test_y) = fashion.load_data()

print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)
plt.imshow(train_x[0],cmap = plt.cm.binary)

plt.show()

print("The Target Variable for Shoes: ",train_y[0])
## For images dataset only and not labels:



train_x = train_x/255

test_x = test_x/255



train_x = train_x.reshape(60000,28,28,1)

test_x = test_x.reshape(10000,28,28,1)

# Reshaping the data to fit the model perfectly



print(train_x.shape)

print(test_x.shape)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=128,activation = tf.nn.relu),

    tf.keras.layers.Dense(units=10, activation = tf.nn.softmax),

    

])



model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',

    metrics = ['accuracy'])
model.fit(train_x,train_y,epochs=3)

test_loss,accuracy = model.evaluate(test_x,test_y)

print(test_loss)

print(accuracy)
predictions = model.predict(test_x)
fashion = tf.keras.datasets.fashion_mnist

(train_x,train_y),(test_x,test_y) = fashion.load_data()

print(test_x.shape)
#### IMPORTANT: For this cell to run test_x should be of the original shape that it came in

## That is 28x28x1 (a grayscale image)





plt.imshow(test_x[0],cmap = plt.cm.binary)

plt.show()



## As 9 is a target variable so,

## if we got 9, it should print an "Ankle Shoe" 

if np.argmax(predictions[0]) == 9:

    print("It is an Ankle Shoe")