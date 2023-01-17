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
train = pd.read_csv('../input/train.csv')
train.shape

test = pd.read_csv('../input/test.csv')
test.shape
X_train = (train.iloc[:, 1:].values).astype('float32')
X_train.shape
y_train = (train.iloc[:, 0].values).astype('int')

y_train.shape
X_test = (test.iloc[:,:].values).astype('float32')
X_test.shape
X_train.shape
X_train = X_train.reshape(-1, 28,28)
X_train.shape
X_test = X_test.reshape(-1, 28,28)
X_test.shape
X_train = X_train.reshape(-1, 28,28,1)
X_train.shape
X_test = X_test.reshape(-1, 28,28,1)
y_train.shape
y_train
import matplotlib.pyplot as plt 
%matplotlib inline 
X_train.shape
y_train.shape
#Feature standardization 

X_train = X_train/ 255.0
X_test = X_test/255.0
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
y_train = to_categorical(y_train, num_classes =10)
y_train.shape
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
random_seed = 2 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10, random_state =random_seed)


X_train.shape
h = plt.imshow(X_train[0] [:, :, 0])
##X_train[0].shape
model = Sequential()
model.add(Conv2D(32, (5,5), padding = "same", activation = "relu", input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "softmax"))





optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

epochs = 5
batch_size = 86
model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,validation_data = (X_val, y_val), verbose =2)
results = model.predict(X_test)
results = np.argmax(results, axis =1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist.csv",index=False)

