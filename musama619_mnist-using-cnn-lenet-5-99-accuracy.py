import keras as kr

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline



import warnings
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv') 
train.shape
test.shape
y_train = train['label']
x_train = train.drop(['label'], axis=1)
x_train.shape
len(y_train.unique()) #0-9 
x_train = x_train.astype('float32')

test = test.astype('float32')
x_train /= 255

test /= 255
plt.imshow(x_train.values.reshape(len(x_train), 28, 28)[9], cmap="Greys")

plt.colorbar()
y_train = kr.utils.to_categorical(y_train) #converting labels to categorized form int form
y_train.shape
x_train = x_train.values.reshape(-1, 28, 28, 1)  #Reshaping to 28x28 grayscale image

test = test.values.reshape(-1, 28, 28, 1)
x_train.shape
test.shape
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten 

from keras import layers
model = Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPool2D(strides=2))

model.add(layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))

model.add(layers.MaxPool2D(strides=2))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(84, activation='relu'))

model.add(layers.Dense(36, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs=25, batch_size=256)
output = model.predict(test)
output = np.argmax(output, axis = 1)
output = pd.Series(output, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), output], axis = 1)
submission.to_csv("submission_05.csv", index=False)