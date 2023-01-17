# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout

from keras.models import Sequential, load_model

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint

data_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sub_file = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
data_train.head()
data_test.head()
x = np.array(data_train.iloc[:, 1:])

y = to_categorical(np.array(data_train.iloc[:, 0]), num_classes=10)
x.shape, y.shape
x = x.reshape(-1, 28, 28, 1)

x.shape
def standardize(data):

    mean = np.mean(data)

    std = np.std(data)

    data_scaled = (data-mean)/std

    return data_scaled



x = standardize(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)
model = Sequential()



model.add(Conv2D(32, (5,5), input_shape=(28,28,1), padding = 'Same', activation='relu'))

model.add(Conv2D(32, (5,5), padding = 'Same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.2))

model.add(Conv2D(64, (5,5), padding = 'Same', activation='relu'))

model.add(Conv2D(64, (5,5), padding = 'Same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (5,5), padding = 'Same', activation='relu'))

model.add(Conv2D(128, (5,5), padding = 'Same', activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(MaxPooling2D(2,2))

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.input_shape
model.output_shape
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=82, epochs=40, validation_data = (x_test, y_test), verbose=2, callbacks=[es, mc])
saved_model = load_model('best_model.h5')
_, train_acc = saved_model.evaluate(x_train, y_train, verbose=0)

train_acc
_, test_acc = saved_model.evaluate(x_test, y_test, verbose=0)

test_acc
data_test.shape
pred_data = np.array(data_test)

pred_data = pred_data.reshape(-1, 28, 28, 1)

pred_data.shape
y_pred = saved_model.predict(pred_data)
y_pred.shape
y_pred = np.argmax(y_pred, axis=1)
y_pred = pd.Series(y_pred, name='Label')
sub_file.head()
y_pred.shape
output = pd.concat([pd.Series(range(1,28001), name='ImageId'), y_pred], axis=1)
output.head()
output.to_csv('results.csv')