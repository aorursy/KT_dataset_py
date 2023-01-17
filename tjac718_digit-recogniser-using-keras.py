#Using the GPU for faster computation
import sys
print(sys.version)
device = 'cuda'
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

from keras import layers
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, ZeroPadding2D, Activation
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.python.client import device_lib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Checking for GPU availability
print(device_lib.list_local_devices())
# Reading the data from the files

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1)

X_train = X_train/255.0
test = test/255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

Y_train = to_categorical(Y_train, num_classes = 10)

random_seed = 1

X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.2, random_state = random_seed)

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization(axis = 3))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization(axis = 1))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
optimizer = Adam(learning_rate = 0.001)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

lr_scheduler = ReduceLROnPlateau(monitor = 'loss', patience = 2, verbose = 1, factor = 0.1, cooldown = 2)

early_stopper = EarlyStopping(monitor = 'loss', patience = 10, verbose = 1)

model.fit(X_train, Y_train, epochs = 150, batch_size = 128, callbacks = [lr_scheduler, early_stopper])

model.evaluate(X_cv, Y_cv, batch_size = 256)
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)