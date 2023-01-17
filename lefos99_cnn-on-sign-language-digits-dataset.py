# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input/Sign-language-digits-dataset/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# load data set

X_l = np.load('../input/Sign-language-digits-dataset/X.npy')

Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')

img_size = 64

plt.subplot(1, 2, 1)

plt.imshow(X_l[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1, 2, 2)

plt.imshow(X_l[900].reshape(img_size, img_size))

plt.axis('off')
print("old X shape: " , X_l.shape)

print("old Y shape: " , Y_l.shape)

X =  np.reshape(X_l, newshape = (X_l.shape[0], X_l.shape[1], X_l.shape[2], 1))

# X = x_l

Y = Y_l

print("new X shape: " , X.shape)

print("new Y shape: " , Y.shape)

num_classes = 10
import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers.normalization import BatchNormalization

import numpy as np



model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



# Split the data

x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, shuffle=True)





datagen = ImageDataGenerator(

    rescale=1./255,

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True)



# compute quantities required for featurewise normalization

# (std, mean, and principal components if ZCA whitening is applied)

# datagen.fit(x_train)



fit_stats = model.fit(x=x_train, y=y_train, batch_size=10, epochs=5, validation_data=(x_valid, y_valid), shuffle=True)
# Visualize training history

print(fit_stats.history.keys())

plt.plot(fit_stats.history['acc'])

plt.plot(fit_stats.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(fit_stats.history['loss'])

plt.plot(fit_stats.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()