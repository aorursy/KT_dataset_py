# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from keras.models import Sequential

from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

from keras.optimizers import Adam



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

subm = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
X_train , X_test , y_train , y_test = train_test_split(train.iloc[: , 1:].values , train.iloc[:,0].values , test_size = 0.1)
X_train = X_train.reshape(-1, 1, 28, 28)/255.

X_test = X_test.reshape(-1, 1, 28, 28)/255.

y_train = np_utils.to_categorical(y_train, num_classes=10)

y_test = np_utils.to_categorical(y_test, num_classes=10)
# Another way to build your CNN

model = Sequential()



# Conv layer 1 output shape (32, 28, 28)

model.add(Convolution2D(

    batch_input_shape=(None, 1, 28, 28),

    filters=32,

    kernel_size=5,

    strides=1,

    padding='same',     # Padding method

    data_format='channels_first',

))

model.add(Activation('relu'))



# Pooling layer 1 (max pooling) output shape (32, 14, 14)

model.add(MaxPooling2D(

    pool_size=2,

    strides=2,

    padding='same',    # Padding method

    data_format='channels_first',

))



# Conv layer 2 output shape (64, 14, 14)

model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))

model.add(Activation('relu'))



# Pooling layer 2 (max pooling) output shape (64, 7, 7)

model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))



# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)

model.add(Flatten())

model.add(Dense(1024))

model.add(Activation('relu'))



# Fully connected layer 2 to shape (10) for 10 classes

model.add(Dense(10))

model.add(Activation('softmax'))



# Another way to define your optimizer

adam = Adam(lr=1e-4)



# We add metrics to get more results you want to see

model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=['accuracy'])
print('Training ------------')

# Another way to train the model

model.fit(X_train, y_train, epochs=1, batch_size=64,)



print('\nTesting ------------')

# Evaluate the model with the metrics we defined earlier

loss, accuracy = model.evaluate(X_test, y_test)



print('\ntest loss: ', loss)

print('\ntest accuracy: ', accuracy)
test = test.iloc[:,1:].values 

test = test.reshape(-1 , 1 , 28 , 28)/255

pre = model.predict(test, batch_size=64)

pre = np.argmax(pre , axis = 1)
subm['label'] = pre

subm.to_csv('submit.csv' , index= False)