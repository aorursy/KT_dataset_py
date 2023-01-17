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
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

h5f = h5py.File('/kaggle/input/street-view-house-nos-h5-file/SVHN_single_grey1.h5', 'r')
# Load the training, test and validation set

x_train = h5f['X_train'][:]

y_train = h5f['y_train'][:]

x_test = h5f['X_test'][:]

y_test = h5f['y_test'][:]





# Close this file

h5f.close()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=5,ncols=5)

fig.figsize=(15,12)

c = 0

for i in range(0,5):

    for j in range(0, 5):

        axes[i,j].imshow(x_test[c,...])

        c+=1
# input image dimensions

img_rows, img_cols = 32, 32



#Keras expects data to be in the format (N_E.N_H,N_W,N_C)

#N_E = Number of Examples, N_H = height, N_W = Width, N_C = Number of Channels.

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



#Normalizing the input

x_train /= 255.0

x_test /= 255.0

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
batch_size = 128

num_classes = 10

epochs = 3 # 12
y_train[:10]
from tensorflow.keras.utils import to_categorical

y_train_ohe = to_categorical(y_train)
y_train_ohe[:10]
y_test_ohe = to_categorical(y_test)

y_test_ohe.shape
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(4,4), input_shape=(32, 32, 1), activation='relu', padding='same', strides=1))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))



model.add(Conv2D(filters = 64, kernel_size=(3,3), activation='relu', padding='valid', strides=1))

model.add(MaxPooling2D(pool_size=(2,2), strides=2))



model.add(Flatten())

model.add(Dense(units=1024, activation='relu'))

model.add(Dropout(rate=0.25))

model.add(Dense(units=10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#model.summary()
%%time

history = model.fit(x_train, y_train_ohe, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.evaluate(x_test, y_test_ohe, verbose=0)
pd.DataFrame(history.history).plot(figsize=(10,6))

plt.grid(True)

plt.show()
history.history