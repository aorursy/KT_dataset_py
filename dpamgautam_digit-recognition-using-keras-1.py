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
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.model_selection import train_test_split
from keras.models import Sequential

from keras.layers import Dense, Dropout, Lambda, Flatten

from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K
train = pd.read_csv("../input/train.csv")

print(train.shape)

(train.head(5))
test = pd.read_csv("../input/test.csv")

print(test.shape)

test.head(5)
x_train = train.iloc[:,1:].values.astype("float32")

y_train = train.iloc[:,0].values.astype("int32")

y_train = y_train.reshape(y_train.shape[0],1)

x_test = test.values.astype("float32")
x_train.shape, y_train.shape, x_test.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28)



for i in range(6,9):

    plt.subplot(330 + (i+1))

    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_train.shape
mean_value = x_train.mean().astype("float32")

std_value = x_train.mean().astype("float32")



def standardize(x):

    return (x-mean_value)/std_value
from keras.utils.np_utils import to_categorical



y_train = to_categorical(y_train)

print(y_train.shape)

num_classes = y_train.shape[1]

print(num_classes)
# plot any label (for ex 10th label)



plt.title(y_train[9])

plt.plot(y_train[9])

plt.xticks(range(10))

plt.show()
# seed for reproducibility



seed = 43

np.random.seed(seed)
from keras.models import Sequential

from keras.layers.core import Dropout, Dense, Lambda, Flatten

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, MaxPooling2D, Convolution2D
model = Sequential()

model.add(Lambda(standardize, input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(10, activation="softmax"))



print("input shape : ", model.input_shape)

print("output shape : ", model.output_shape)
from keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
from keras.preprocessing import image

gen = image.ImageDataGenerator()
from sklearn.model_selection import train_test_split

x = x_train

y = y_train

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)



batches = gen.flow(x_train, y_train, batch_size=64)

val_batches = gen.flow(x_val, y_val, batch_size=64)
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3, validation_data=val_batches, validation_steps=val_batches.n)
history_dict = history.history

history_dict
import matplotlib.pyplot as plt

%matplotlib inline



loss_values = history_dict["loss"]

val_loss_values = history_dict["val_loss"]

epochs = range(1, len(loss_values)+1)



plt.plot(epochs, loss_values, "-bo")

plt.plot(epochs, val_loss_values, "-b+")

plt.xlabel("Epochs")

plt.ylabel("Losses")

plt.show()
acc_values = history_dict["acc"]

val_acc_values = history_dict["val_acc"]

plt.plot(epochs, acc_values, "-ro")

plt.plot(epochs, val_acc_values, "-r+")

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.show()
def get_fc_model():

    model = Sequential([Lambda(standardize, input_shape=(28,28,1)),Flatten(),Dense(512, activation='relu'),Dense(10, activation='softmax')])

    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

    return model



fc = get_fc_model()

fc.optimizer.lr=0.01
history = fc.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)
from keras.layers import Convolution2D, MaxPooling2D
def get_cnn_model():

    model = Sequential( [Lambda(standardize, input_shape=(28,28,1)), Convolution2D(32,(3,3), activation="relu"), Convolution2D(32,(3,3), activation='relu'), MaxPooling2D(), Convolution2D(64,(3,3), activation='relu'), Convolution2D(64,(3,3), activation='relu'), MaxPooling2D(), Flatten(), Dense(512, activation='relu'), Dense(10, activation='softmax') ] )

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = get_cnn_model()

model.optimizer.lr=0.01
history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)
gen = ImageDataGenerator( rotation_range=8, width_shift_range=0.08, shear_range=0.3, height_shift_range=0.08, zoom_range=0.08 )

batches = gen.flow(x_train, y_train, batch_size=64)

val_batches = gen.flow(x_val, y_val, batch_size=64)
model.optimizer.lr=0.001

history = model.fit_generator( generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n )
from keras.layers.normalization import BatchNormalization
def get_bn_model():

    model = Sequential( [ Lambda(standardize, input_shape=(28,28,1)), Convolution2D(32,(3,3), activation='relu'),BatchNormalization(axis=1),Convolution2D(32,(3,3), activation='relu'),MaxPooling2D(),BatchNormalization(axis=1),Convolution2D(64,(3,3), activation='relu'),BatchNormalization(axis=1),Convolution2D(64,(3,3), activation='relu'),MaxPooling2D(),Flatten(),BatchNormalization(),Dense(512, activation='relu'),BatchNormalization(),Dense(10, activation='softmax') ] )

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
model = get_bn_model()

model.optimizer.lr=0.01

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1, validation_data=val_batches, validation_steps=val_batches.n)
model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X, y, batch_size=64)

history=model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3)