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
from tensorflow.keras.datasets import mnist



(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train shape: ', x_train.shape)

print('y_train shape: ', y_train.shape)

print('x_test shape: ', x_test.shape)

print('y_test shape: ', y_test.shape)
import matplotlib.pyplot as plt

%matplotlib inline



plt.imshow(x_train[0], cmap = 'Greens')

plt.show()
y_train[0]
y_train[:10]
import tensorflow



y_train_encoded = tensorflow.keras.utils.to_categorical(y_train)

y_test_encoded = tensorflow.keras.utils.to_categorical(y_test)
print('y_train shape: ', y_train_encoded.shape)

print('y_test shape: ', y_test_encoded.shape)
y_train_encoded[0]
import numpy as np



x_train_reshaped = np.reshape(x_train, (60000, 784))

x_test_reshaped = np.reshape(x_test, (10000, 784))



print('x_train_reshaped shape: ', x_train_reshaped.shape)

print('x_test_reshaped shape: ', x_test_reshaped.shape)
print(set(x_train_reshaped[0]))
x_mean = np.mean(x_train_reshaped)

x_std = np.std(x_train_reshaped)



print('mean: ', x_mean)

print('std: ', x_std)
epsilon = 1e-10

#we are using episolon so as to avoid any instability due to x_std

x_train_norm = (x_train_reshaped - x_mean)/(x_std + epsilon)

x_test_norm = (x_test_reshaped - x_mean)/(x_std + epsilon)
print(set(x_train_norm[0]))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense



model = Sequential()

model.add(Dense(units=128, activation = 'relu', input_dim=784,kernel_initializer='he_uniform'))

model.add(Dense(units=128, activation = 'relu',kernel_initializer='he_normal'))

model.add(Dense(units=10, activation = 'softmax',kernel_initializer='normal'))
model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics = ['accuracy'])

model.summary()
h = model.fit( x_train_norm,y_train_encoded,epochs = 5,validation_split=0.33,batch_size=10)

loss, accuracy = model.evaluate(x_test_norm, y_test_encoded)

print('test set accuracy: ', accuracy * 100)

y_pred = model.predict(x_test_norm)

print('shape of predictions: ', y_pred.shape)

plt.figure(figsize = (12, 12))

t=0

for i in range(30,60):

    plt.subplot(6, 5, t + 1)

    plt.grid(False)

    plt.xticks([])

    plt.yticks([])

    col = 'g'

    if np.argmax(y_pred[i])!= np.argmax(y_test_encoded[i]):

        col = 'r'

    plt.xlabel('i={},p_val={},t_val={}'.format(i, np.argmax(y_pred[i]),np.argmax(y_test_encoded[i]) ), color = col)

    plt.imshow(x_test[i], cmap='Reds')

    t=t+1

plt.show()