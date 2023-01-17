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
train_images=pd.read_csv('../input/regularization-techniques/train.csv')
test_images=pd.read_csv('../input/regularization-techniques/test.csv')
train_images.head()
test_images.head()
x_train = train_images.drop('label', axis=1).to_numpy()
x_train = x_train.reshape(-1, 784).astype('float32')
y_train = train_images['label'].to_numpy()
x_test = test_images.drop('label', axis=1).to_numpy()
x_test = x_test.reshape(-1, 784).astype('float32')
y_test = test_images['label'].to_numpy()
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
model = Sequential()
model.add(Dense(units = 500, input_dim=784, activation='relu',activity_regularizer=regularizers.l1(1e-4))) # Hidden 1
model.add(Dense(units = 500, activation='relu', activity_regularizer=regularizers.l1(1e-4))) # Hidden 2
model.add(Dense(units = 1,activation='softmax')) # Output
model.compile(loss='categorical_crossentropy', optimizer='adam')

trained_model = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=128)
model = Sequential([
 Dense(units=500, input_dim=784, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
 Dense(units=500, input_dim=784, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
 Dense(units=1, input_dim=500, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
from keras.layers.core import Dropout
model = Sequential([
 Dense(units=500, input_dim=784, activation='relu'),
 Dropout(0.25),
 Dense(units=500, input_dim=500, activation='relu'),
 Dropout(0.25),
 Dense(units=1, input_dim=500, activation='softmax'),
 ])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
from keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
trained_model_5d = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), callbacks = [EarlyStopping(monitor='val_acc', patience=2)])