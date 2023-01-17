import keras

import pandas as pd

import numpy as np

from keras.datasets import mnist

import matplotlib.pyplot as plt

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(train.shape)

train.head()
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test.head()
x_train = train.iloc[:,1:].values

y_train = train.iloc[:,0].values

x_test = test.values
plt.figure()

fig, ax = plt.subplots(1,6)

for i in range(0,6):

  ax[i].imshow(x_train[i].reshape(28,28))
print(x_train.max())

print(x_train.min())
x_train = x_train.reshape(x_train.shape[0],28,28,1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
x_train = x_train.astype('float32')

x_train = x_train/255

x_test = x_test.astype('float32')

x_test = x_test/255
y_train = to_categorical(y_train,10)

#y_test = to_categorical(y_test,10)
model = Sequential()

# input_shape takes tuple with (height, width, channels) only for first layer

model.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=x_train.shape[1:]))

model.add(Conv2D(filters=32, kernel_size=5, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
epochs = 1

batch_size = 32

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
y_predict = model.predict_classes(x_test)

np.savetxt("output.csv",np.c_[range(1,len(y_predict)+1), y_predict], delimiter=',', header="ImageID,Label", comments = '', fmt='%d')