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
import tensorflow as tf

import tensorflow.keras as keras

data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')

for i in range(23):

    data.iloc[:,i]=data.iloc[:,i].astype('category').cat.codes

x_train = data.iloc[:8000,1:]

y_train = data.iloc[:8000,:1]

y_train["class2"]=1-y_train["class"]



model = tf.keras.Sequential()

model.add(keras.layers.Dense(32,input_dim=22))



model.add(keras.layers.Dense(32,activation=tf.nn.relu))



model.add(keras.layers.Dense(32,activation=tf.nn.sigmoid))



model.add(keras.layers.Dense(16,activation=tf.nn.relu))



model.add(keras.layers.Dense(2,activation=tf.nn.softmax))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.fit(x_train,y_train,validation_split=0.05,epochs=30)



x_test = data.iloc[8000:,1:]

y_test = data.iloc[8000:,:1]

y_test["class2"]=1-y_test["class"]

test_loss, test_acc = model.evaluate(x_test, y_test)



print('Test accuracy:', test_acc)

model.save('my_model_final.h5')