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

import sklearn

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf



import math

import os,datetime

from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

dataset = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
# Continuing with the EDA Analysis as done by other contributors #

data_x = dataset.iloc[:,1:].values

data_y = dataset.iloc[:,0].values



data_y = np.reshape(data_y,(data_y.shape[0],1))
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()

for i in range(0,data_x.shape[1]):

    data_x[:,i] = labelencoder.fit_transform(data_x[:,i])

    

labelencoder = LabelEncoder()

data_y[:,0] = labelencoder.fit_transform(data_y[:,0])
onehotencoder= OneHotEncoder()

data_x=onehotencoder.fit_transform(data_x).toarray()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size = 0.3,random_state=0)

print("x_train"+str(x_train.shape))

print("x_test"+str(x_test.shape))

print("y_train"+str(y_train.shape))

print("y_test"+str(y_test.shape))
n_x = x_train.shape[1]

x_train = np.float32(x_train)

x_test = np.float32(x_test)

y_train = np.float32(y_train)

y_test = np.float32(y_test)

tf.random.set_seed(1)

model  = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(5,activation = 'relu', kernel_initializer = 'glorot_normal' , bias_initializer = 'glorot_normal',input_shape = (n_x,)))

model.add(tf.keras.layers.Dense(3, activation='relu', kernel_initializer='glorot_normal',bias_initializer = 'glorot_normal',name = "relu2"))

model.add(tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer = 'glorot_normal',bias_initializer = 'glorot_normal',name="output"))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy','mean_squared_error'])
training_history = model.fit(x_train,y_train,epochs=7,batch_size = 16,verbose=1,use_multiprocessing=True)





losstrain,acctrain,msetrain = model.evaluate(x_train,y_train,verbose=0)

loss, acc,msetest = model.evaluate(x_test, y_test, verbose=0)







print('Train Accuracy: %.3f' % acctrain)

print('Test Accuracy: %.3f' % acc)

print("Average test loss: ", np.average(training_history.history['loss']))
intermediate_layer_model = Model(inputs=model.input,

                                 outputs=model.get_layer('output').output)

intermediate_output = intermediate_layer_model.predict(x_train)



print(intermediate_output)
plt.title("LOSS")

plt.plot(training_history.history['loss'])

print(training_history.history)

plt.title("MSE")

plt.plot(training_history.history['mean_squared_error'])