# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from keras import backend as k

from keras.utils import np_utils

from keras.layers import Dense, Flatten, Dropout, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.layers.core import Activation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

test_data_copy = pd.read_csv('../input/test.csv')

ntrain = train_data.shape[0]

y = train_data.iloc[:,0]

train_data = train_data.drop(['label'],axis=1)
print(train_data.shape)

print(test_data.shape)

train_data.head()
'''

total_data = train_data.append(test_data)

total_data = total_data/255

'''

train_data = train_data/255

test_data = test_data/255

# y label counts

unique,counts= np.unique(y, return_counts=True)

print([unique,counts])



# we can see all classes have similar number of training examples
x_train, x_test, y_train, y_test = train_test_split(train_data, y.values, test_size=0.2 , random_state=42)



x_train = np.array(x_train.values).reshape(x_train.shape[0], 28, 28 , 1).astype('float32')

x_test = np.array(x_test.values).reshape(x_test.shape[0], 28, 28 , 1).astype('float32')

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(filters= 32, kernel_size=5, input_shape=(28,28,1), activation='relu', padding='same'))

model.add(Conv2D(filters= 32, kernel_size=5, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Conv2D(filters= 64, kernel_size=3, activation='relu', padding='same'))

model.add(Conv2D(filters= 64, kernel_size=3, activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(3,3)))

model.add(BatchNormalization())

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(units=64, activation="relu"))

model.add(BatchNormalization())

model.add(Dense(units=128, activation="relu"))

model.add(BatchNormalization())

model.add(Dense(units=256, activation="relu"))

model.add(Dropout(0.4))

model.add(BatchNormalization())

model.add(Dense(10))

model.add(Activation("softmax"))
optimizer = RMSprop(lr=0.0002500000118743628, rho=0.9, epsilon=1e-08, decay=0.0)



# 0.0002500000118743628

#0.001
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=optimizer)

# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
datagen = ImageDataGenerator(

                        rotation_range = 10,

                        zoom_range = 0.1, # Randomly zoom image 

                        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

)

datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=100), epochs=10, steps_per_epoch=1000, verbose=1,validation_data = (x_test, y_test))

                              #,callbacks = [learning_rate_reduction])

#predictin test_data

test_data1 = np.array(test_data.values).reshape(test_data.shape[0], 28, 28 , 1).astype('float32')

y_pred = model.predict(test_data1)
y_pred = np.argmax(y_pred,axis=1)

y_pred
solutions = pd.DataFrame({'ImageId':pd.Series(range(1 ,28001)), 'Label':y_pred})

solutions.to_csv('digit_1.csv',index=False)