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
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

data_train = pd.read_csv('../input/train.csv')
x_test = pd.read_csv('../input/test.csv')
X_train = np.array(data_train[['pixel' + str(i) for i in range(0,784)]])
y_train = np.array(data_train['label'])
X_test = np.array(x_test[['pixel' + str(i) for i in range(0,784)]])
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (32,32,1)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu',input_shape = (14,14,6)))
#Pooling Layer 2
model.add(MaxPooling2D(pool_size = 2, strides = 2))
#Flatten
model.add(Flatten())
#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120, activation = 'relu'))
#Layer 4
#Fully connected layer 2
model.add(Dense(units = 84, activation = 'relu'))
#Layer 5
#Output Layer
model.add(Dense(units = 10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train ,y_train, steps_per_epoch = 10, epochs = 42)
y = model.predict(X_test)
y = np.argmax(y, axis = 1)
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(x_test)+1)]
submission['Label'] = y
submission.to_csv('submission.csv', index=False)