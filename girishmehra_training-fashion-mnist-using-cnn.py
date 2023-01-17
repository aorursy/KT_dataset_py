# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.models import Sequential
from keras.layers import Input,Convolution2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.utils import np_utils
import tensorflow
x = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
X_ = np.array(x)
X = X_[:,1:]
X = X/255.0
y = X_[:,0]
print(X.shape,y.shape)
print(y[:100])
x.head()
np.unique(y,return_counts=True)
X_train = X.reshape((-1,28,28,1))
Y_train = np_utils.to_categorical(y)


print(X_train.shape,Y_train.shape)
import matplotlib.pyplot as plt
for i in range(10):
    plt.imshow(X_train[i].reshape(28,28),cmap = 'gray')
    plt.show()
model = Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape = (28,28,1)))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(2,2))
model.add(Convolution2D(32,(5,5),activation='relu'))
model.add(Convolution2D(8,(5,5),activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
hist = model.fit(X_train,Y_train,epochs=10,batch_size=256,shuffle=True,validation_split=0.20)
plt.figure(0)
plt.plot(hist.history['loss'],'g')
plt.plot(hist.history['val_loss'],'b')
plt.plot(hist.history['accuracy'],'r')
plt.plot(hist.history['val_accuracy'],'black')
plt.show()
