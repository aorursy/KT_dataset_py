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
import keras
from keras.datasets import mnist # 28x28 image data written digits 0-9
from keras.utils import normalize

#print(keras.__version__)

#split train and test dataset 
(x_train, y_train), (x_test,y_test) = mnist.load_data()

#normalize data 
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)
import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()
#print(x_train[0])
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense 

# created model 
model = Sequential()

# flatten layer so it is operable by this layer  
model.add(Flatten())

# regular densely-connected NN layer.
#layer 1, 128 node 
model.add(Dense(128, activation='relu'))

#layer 2, 128 node 
model.add(Dense(128, activation='relu'))

#output layer, since it is probability distribution we will use 'softmax'
model.add(Dense(10, activation='softmax'))
from keras.callbacks import EarlyStopping

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#stop when see model not improving 
early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x_train, y_train, epochs=10, callbacks=[early_stopping_monitor], validation_data=(x_test, y_test))
val_loss, val_acc = model.evaluate(x_test,y_test)
print(val_loss, val_acc)
model.save('mnist_digit.h5')
model.summary()
from keras.models import load_model

new_model = load_model('mnist_digit.h5')
predict = new_model.predict([x_test])

#return the probability 
print(predict)
print(predict[1].argmax(axis=-1))
plt.imshow(x_test[1])
plt.show()