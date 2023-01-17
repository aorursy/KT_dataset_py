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
# for visual data

import matplotlib.pyplot as plt

import random

%matplotlib inline



# for deep learning

from tensorflow.keras.layers import Dense, Flatten, Activation,Dropout,Conv2D, MaxPooling2D

from tensorflow.keras.models import Sequential

from keras.utils import np_utils

from tensorflow.keras.callbacks import ModelCheckpoint
# first import the data

training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
# take look on the raw data

training_data.head()

# slice the label to y and other columns to x

Y_train = training_data['label']

X_train = np.asarray(training_data.loc[:,"pixel0":])/255 # slice and scale

X_test = np.asarray(testing_data)/255 # scale
# reshape the data

X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)
X_train.shape
# split the train data into train and valid data we need only 0.2 of data be valid

X_tarin, X_valid = X_train[int(X_train.shape[0]*0.2):], X_train[ : int(X_train.shape[0]*0.2)]

Y_tarin, Y_valid = X_train[int(Y_train.shape[0]*0.2):], Y_train[ : int(X_train.shape[0]*0.2)]

# print number of training, validation, and test images

print(X_train.shape[0], 'train samples')

print(X_test.shape[0], 'test samples')

print(X_valid.shape[0], 'validation samples')


# visual random image

plt.imshow(random.choice(X_train).reshape(28,28),cmap='gray')
model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 

                        input_shape=(28, 28, 1)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))



model.add(Dense(10, activation='softmax'))
# compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 

# train the model and save the best accuracy

checkpoint_path = "Digit_Recognizer.hdf5" 

checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=1, 

                               save_best_only=True)



best_model = model.fit(X_train, Y_train, batch_size=32, epochs=20,

          validation_data=(X_valid, Y_valid),  verbose=1, shuffle=True, callbacks=[checkpointer])
results = model.predict(X_test)
fig = plt.figure(figsize=(20, 8))

for i, idx in enumerate(np.random.choice(X_test.shape[0], size=24, replace=False)): 

    ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(X_test[idx]),cmap='gray')

    pred_idx = np.argmax(results[idx])

    ax.set_title("{}".format(pred_idx))
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)

submission.head()
