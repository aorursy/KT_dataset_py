# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
X = np.load("../input/Sign-language-digits-dataset/X.npy")
Y = np.load("../input/Sign-language-digits-dataset/Y.npy")
print(X.shape)
print(Y.shape)
X = X.reshape(-1,64,64,1)
print(X.shape)
X = X.reshape(-1,64,64,1)
print(X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15, random_state=42)
print(X_test.shape)
plt.imshow(X_train[120][:,:,0])
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (4,4),padding = 'Same', 
                 activation ='relu', input_shape = (64,64,1)))
model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(4,4)))
model.add(MaxPooling2D(pool_size=4))
#
model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', 
                 activation ='relu'))
model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(4,4), strides=(4,4)))
model.add(MaxPooling2D(pool_size=4))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dense(10, activation = "softmax"))
# model = Sequential()

# # Convolutional Blocks: (1) Convolution, (2) Activation, (3) Pooling
# model.add(Conv2D(activation ='relu', input_shape=(64, 64, 1), filters=64, kernel_size=(4,4), strides=(2)))
# #outputs a (20, 20, 32) matrix
# model.add(Conv2D(activation ='relu', filters=64, kernel_size=(4,4), strides=(1)))
# #outputs a (8, 8, 32) matrix
# model.add(MaxPooling2D(pool_size=4))

# # dropout helps with over fitting by randomly dropping nodes each epoch
# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())

# model.add(Dense(10, activation='softmax'))
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 50  # for better result increase the epochs
batch_size = 128

datagen = ImageDataGenerator(
         rotation_range=16,
         width_shift_range=0.12,
         height_shift_range=0.12,
         zoom_range=0.12)  # randomly flip images

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs)
score = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy is:",score[1])
plt.imshow(X_test[20][:,:,0])
test_image = X_test[20]
print(test_image.shape)
test_image_array = test_image.reshape(64, 64)
print(test_image_array.shape)
test_image = np.expand_dims(test_image, axis = 0)
print(test_image.shape)
result = model.predict(test_image)
print(result)
print(np.round(result, 1))
print(Y_test[20])
