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
train_data = pd.read_csv('../input/train.csv').values
test_data = pd.read_csv('../input/test.csv').values
X_train = train_data[:,1:]
X_train = np.array(X_train)
Y_train = train_data[:,[0]]
Y_train = np.array(Y_train)

X_test = test_data[:,:]
X_test = np.array(X_test)
Y_test = test_data[:,[0]]
Y_test = np.array(Y_test)
X_train_deflatten = np.zeros((42000,28,28,1))
for image in range(0,42000):
    for pixel in range(0,784):
        X_train_deflatten[image, pixel%28, (int)((pixel-pixel%28)/28)] = X_train[image, pixel]
X_test_deflatten = np.zeros((28000, 28, 28,1))
for image in range(0,28000):
    for pixel in range(0,784):
        X_test_deflatten[image, pixel%28, (int)((pixel-pixel%28)/28),0] = X_test[image, pixel]
Y_categorical = np.zeros((42000,10))
for image in range(0,42000):
    for category in range(0,10):
        Y_categorical[image, Y_train[image]] = 1
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
classificator = Sequential()
classificator.add(Conv2D(8, (3, 3), input_shape=(28, 28, 1),activation='relu'))
classificator.add(MaxPooling2D(pool_size=(2,2)))
classificator.add(Conv2D(8, (3, 3), activation='relu'))
classificator.add(MaxPooling2D(pool_size=(2,2)))
classificator.add(Flatten())
classificator.add(Dense(units=32, activation="relu"))
classificator.add(Dense(units=10, activation="sigmoid"))

classificator.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['mean_squared_error'])
classificator.fit(X_train_deflatten, Y_categorical,steps_per_epoch = 1,epochs = 1)










from keras.models import load_model

classificator.save('my_model.h5')
prediction = classificator.predict(X_test_deflatten)
print(prediction)
def indexOfHighest(x):
    maximum = 0
    maxIndex = -1
    for i in range(0,x.size):
        if(x[i]>maximum):
            maximum = x[i]
            maxIndex = i
    return maxIndex

print(indexOfHighest(prediction[0,:]))
print(Y_test[0])
print(X_test_deflatten[0,:,::2,0])
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import numpy as np
import matplotlib.pyplot as plt

H = X_test_deflatten[0,:,:,0]

fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(H)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()






















