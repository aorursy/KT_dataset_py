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
#Thanks to rahuldshetty for data preprocessing steps.

#CNN model needs to be finetuned

#https://www.kaggle.com/rahuldshetty/kannada-digits-mnist-using-cnn-0-97



import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

import gc

from datetime import datetime



from sklearn.model_selection import train_test_split

from keras.utils import to_categorical 

from keras import backend as K

from keras.layers import Dense, Dropout,Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Sequential

from keras.layers import Dropout

from keras.layers.core import Activation

from keras.layers.normalization import BatchNormalization
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

valid = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
train.head()
test.head()
train_arr = train.values[0:,1:]

labels = train.values[0:,0]

labels = to_categorical(labels)
plt.imshow(train_arr[0].reshape((28,28)))

print(f'this is a {labels[0][0]}')
train_arr = train_arr/255.0

train_arr = train_arr.reshape(train_arr.shape[0], 28, 28, 1)
valid_arr = valid.values

valid_arrx = valid_arr[0:,1:]

valid_arry = valid_arr[0:,0]
valid_arrx = valid_arrx/255.0

valid_arrx = valid_arrx.reshape(valid_arr.shape[0], 28, 28, 1)

valid_arry = to_categorical(valid_arry)
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=train_arr.shape[1:], activation='relu', padding='same'))

# model.add(Conv2D(32,(3,3), activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.2))



model.add(Conv2D(64,(3,3), activation='relu', padding='same'))

# model.add(Conv2D(64,(3,3), activation='relu'))

model.add(MaxPooling2D((2,2)))

model.add(Dropout(0.2))



# model.add(Conv2D(128,(3,3), activation='relu', padding='same'))

# # model.add(Conv2D(128,(3,3), activation='relu'))

# model.add(MaxPooling2D((2,2)))

# model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()
history = model.fit(train_arr, labels, validation_data=(valid_arrx,valid_arry), epochs=15, batch_size=128)
test = test.values

test_arr = test[0:,1:]

id_samples = test[0:,0]

test_arr = test_arr / 255.0

test_arr = test_arr.reshape((test_arr.shape[0],28,28,1))
y_pred = model.predict_classes(test_arr)
sub = pd.DataFrame()

sub['id'] = list(id_samples)

sub['label'] = y_pred

sub.head()

sub.to_csv('submission.csv',index = False)