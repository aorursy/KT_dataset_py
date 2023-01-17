# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

seed = 2
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
url_train = '../input/train.csv'
url_test = '../input/test.csv'
train = pd.read_csv(url_train)
test = pd.read_csv(url_test)
x_train = train.iloc[:,1:]
y_train = train.iloc[:,0]
del train
x_train.shape
x_train.head()
y_train.shape
y_train.head()
x_train = x_train/255.0
test = test/255.0
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Would've 3 at the end if RGB image
#Label Encoding
y_train = to_categorical(y_train, num_classes = 10)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = seed)
#Defining CNN layers

# Architecture: In -> [[Conv2D -> relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#Defining Optimizer
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
epochs = 3
batch_size = 86*2
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val), verbose = 2)
predictions = model.predict(test)
#predictions[0]
#np.argmax(predictions[0])
y_pred = np.argmax(predictions, axis = 1)
#y_pred.shape
#y_pred[:50]
sub = pd.DataFrame()
sub['ImageID'] = np.arange(1, 28001, 1)
#sub['ImageID'] = sub['ImageID'] + 1
#sub2 = pd.DataFrame()
#sub2['Label'] = y_pred
sub['Label'] = y_pred
sub.tail()
sub.to_csv('submission.csv', index=False)
