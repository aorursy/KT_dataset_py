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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras.utils import to_categorical

import matplotlib.pyplot as plt



import h5py



%matplotlib inline
with h5py.File('../input/3d-mnist/full_dataset_vectors.h5', 'r') as hf:

    x_train_ = hf["X_train"][:]

    y_train_ = hf["y_train"][:]

    x_test_ = hf["X_test"][:]

    y_test_ = hf["y_test"][:]
# 1D vector to rgb values, provided by ../input/plot3d.py

def array_to_color(array, cmap="Oranges"):

    s_m = plt.cm.ScalarMappable(cmap=cmap)

    return s_m.to_rgba(array)[:,:-1]



# Transform data from 1d to 3d rgb

def rgb_data_transform(data):

    data_t = []

    for i in range(data.shape[0]):

        data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))

    return np.asarray(data_t, dtype=np.float32)

n_classes = 10 # from 0 to 9, 10 labels totally



x_train = rgb_data_transform(x_train_)

x_test = rgb_data_transform(x_test_)



y_train = to_categorical(y_train_, n_classes)

y_test = to_categorical(y_test_, n_classes)
print('\nTrain')

unique_elements, counts_elements = np.unique(y_train_, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))



print('\nTest')

unique_elements, counts_elements = np.unique(y_test_, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
# !pip install keras-rectified-adam
from keras.models import Sequential

from keras.layers import Conv3D, MaxPool3D, Dense, Flatten, Dropout, BatchNormalization, Activation

from keras.optimizers import Adadelta, Adam, RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras_radam import RAdam
filepath="best_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')

early_stopping = EarlyStopping(patience = 10,monitor='val_loss', verbose=0, mode='min')

callbacks_list = [checkpoint, early_stopping]
models= Sequential()

models.add(Dense(256, input_shape=(4096,)))

models.add(BatchNormalization())

models.add(Activation('elu'))

models.add(Dropout(0.25))



models.add(Dense(256))

models.add(BatchNormalization())

models.add(Activation('elu'))

models.add(Dropout(0.25))



models.add(Dense(256))

models.add(BatchNormalization())

models.add(Activation('elu'))

models.add(Dropout(0.25))



models.add(Dense(128))

models.add(BatchNormalization())

models.add(Activation('elu'))

models.add(Dropout(0.5))



models.add(Dense(10, activation='softmax'))
models.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0008), metrics=['acc'])

models.fit(x_train_, y_train_, batch_size=32, epochs=150, validation_split=0.2, verbose=2,\

          callbacks=callbacks_list)
models.evaluate(x_test_,y_test_,batch_size=32)
model = Sequential()



model.add(Conv3D(64,(3,3,3), input_shape=(16,16,16,3), padding='same'))

model.add(Activation('relu'))

model.add(Dropout(0.25))



model.add(MaxPool3D(pool_size=(2, 2, 2)))



# model.add(Conv3D(32,(3,3,3), padding='same'))

# model.add(Activation('relu'))



# model.add(MaxPool3D(pool_size=(2, 2, 2)))



model.add(Flatten())



# model.add(Dense(256))

# model.add(Activation('relu'))



# model.add(Dense(256))

# model.add(Activation('relu'))

# model.add(Dropout(0.25))



model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['acc'])

#model.compile(RAdam(),loss='categorical_crossentropy', metrics=['acc'])

hist=model.fit(x_train, y_train, batch_size=64, epochs=80, validation_split=0.01, verbose=2)
model.evaluate(x_test,y_test,batch_size=64)
loss = hist.history['loss']

val_loss = hist.history['val_loss']



acc = hist.history['acc']

val_acc = hist.history['val_acc']
fig, ax = plt.subplots(2,1, figsize=(12, 12))

ax[0].plot((loss), 'bo', label="Loss")

ax[0].plot((val_loss), 'b', label="Valid_Loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)

ax[0].set_xlabel('Epochs')

ax[0].set_ylabel('Loss')



ax[1].plot((acc), 'bo', label="Accuracy")

ax[1].plot((val_acc), 'b',label="Valid_Accuracy")

legend = ax[1].legend(loc='best', shadow=True)

ax[1].set_xlabel('Epochs')

ax[1].set_ylabel('Accuracy')