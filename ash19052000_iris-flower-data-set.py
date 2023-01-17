# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import sklearn

from sklearn.preprocessing import LabelEncoder



# Reading Data and cleaning data

iris = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

iris_sh = iris.sample(n=150,random_state = 1)

iris_deta = iris_sh[['sepal_length','sepal_width','petal_length','petal_width']]

iris_spc = iris_sh[['species']]

iris
# train/test split

train_dat = iris_deta[0:120]

y1 = iris_spc[0:120]



encoder = LabelEncoder()

train_labels = encoder.fit_transform(y1)

#= pd.get_dummies(y3).values (for converting it in binary)



test_dat = iris_deta[120:150]

y2 = iris_spc[120:150]



test_labels = encoder.fit_transform(y2)



x = train_dat.values #returns a numpy array

min_max_scaler = sklearn.preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

train_data = pd.DataFrame(x_scaled)





y = test_dat.values #returns a numpy array

min_max_scaler1 = sklearn.preprocessing.MinMaxScaler()

y_scaled = min_max_scaler1.fit_transform(y)

test_data = pd.DataFrame(y_scaled)

# pd.get_dummies(y4).values  (for converting it in binary)



#train_labels

#test_labels

#y2

train_data
class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
train_data.shape
test_data.loc[0]
model = keras.Sequential([

  

    keras.layers.Dense(512, activation='relu',input_shape=(4,)),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(512, activation='relu'),

    keras.layers.Dense(3,activation='softmax')

])
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=100,validation_data=(test_data,test_labels))
test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)



print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model])
predictions = probability_model.predict(test_data)
predictions