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
from __future__ import print_function

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau

batch_size = 128

num_classes = 10

epochs = 25

img_rows, img_cols = 28, 28
data_train = pd.read_csv("../input/train.csv")



x_train = (data_train.iloc[:,1:].values).astype('float32')

y_train = data_train.iloc[:,0].values.astype('int32')



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

print('train data shape: ' + str(x_train.shape))

print('validation data shape: ' + str(x_val.shape))
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')

x_val = x_val.astype('float32')

x_train /= 255

x_val /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)

y_val = keras.utils.to_categorical(y_val, num_classes)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1,

                              patience=2, min_lr=0.00000001)
model.fit(x_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_val, y_val), callbacks=[reduce_lr])
data_test= pd.read_csv("../input/test.csv")

x_test = data_test.values.astype('float32')



x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)



x_test = x_test.astype('float32')

x_test /= 255

print('test data shape: ' + str(x_test.shape))
preds = model.predict_classes(x_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(preds)+1)),

                         "Label": preds})

submissions.to_csv("./submission.csv", index=False, header=True)

submissions.head()