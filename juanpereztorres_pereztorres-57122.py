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
# Ejecutar esta celda si quieren usar MNIST
from keras.datasets import fashion_mnist
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Descripción de cada una de las clases
label_dict = {
                0:'T-shirt/top',
                1:'Trouser',
                2:'Pullover',
                3:'Dress',
                4:'Coat',
                5:'Sandal',
                6:'Shirt',
                7:'Sneaker',
                8:'Bag',
                9:'Ankle boot'
            }
y_test[0:10]
x_train.shape
x_test.shape
x_train_reshaped = x_train.reshape(60000,28,28,1)
x_test_reshaped = x_test.reshape(10000,28,28,1)
from matplotlib import pyplot as plt

def create_row(x_train, numbers):
    concatenated = x_train[numbers[0]]
    numbers=numbers[1:]
    for n in numbers:
        concatenated = np.concatenate((concatenated, x_train[n]), axis=1)
    return concatenated

def plot_numbers(x_train, numbers, columns=10, show_label=True, figsize=(20, 5)):
    plt.figure(figsize=figsize)
    numbers = np.array(numbers).reshape(-1, columns)
    concatenated = create_row(x_train, numbers[0])
    numbers = numbers[1:,:]
    for row in numbers:
        concatenated = np.concatenate((concatenated, create_row(x_train, row)))
    plt.imshow(concatenated, cmap='gray')
    plt.show()
N_to_plot = 150
plot_numbers(x_train.reshape(-1,28,28),range(N_to_plot), columns=25, figsize=(20, 20))
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
model = Sequential()
model.add(Conv2D(32,kernel_size=3, activation="relu",input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))
model.add(Conv2D(64,kernel_size=3, activation='relu', padding='same'))
model.add(Conv2D(64,kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer=Adam(lr=0.001), metrics=["acc"])
checkpointer = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(x_train_reshaped,
                    y_train, 
                    validation_split=0.2, 
                    batch_size=256, 
                    epochs=40, 
                    callbacks= [checkpointer])
plt.plot(history.history["acc"],label="train acc")
plt.plot(history.history["val_acc"], label="val acc")
plt.legend()
plt.show()
plt.plot(history.history["loss"],label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.show()
model.evaluate(x_test_reshaped,y_test)
model.predict(x_test_reshaped[0:3]).argmax(axis=1)