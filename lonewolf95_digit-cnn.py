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
import numpy as np

import pandas as pd

import time

import matplotlib.pyplot as plt

#import warnings; warnings.simplefilter('ignore')

%env JOBLIB_TEMP_FOLDER=/tmp

import seaborn as sns

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

import seaborn as sns

%matplotlib inline 
x_train = pd.read_csv('../input/train.csv')

y_train = x_train['label'].values

x_train = x_train.drop('label', axis = 1).values

test    = pd.read_csv('../input/test.csv').values
x_train = (x_train - x_train.mean())/x_train.std()

test = (test - test.mean())/test.std()
print(x_train.shape)

print(y_train.shape)
x_train  =  x_train.reshape((x_train.shape[0],28,28,1))

test  =  test.reshape((test.shape[0],28,28,1))

print(x_train.shape)

print(test.shape)
plt.imshow(x_train[4][:,:,0])
y_train = to_categorical(y_train)
model = Sequential()

model.add(Conv2D(128, (5,5), padding='same', input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, (7, 7), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(192, (3, 3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, (3, 3), padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(1, 1)))



model.add(Flatten())

model.add(Dense(4096))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dense(4096))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dense(10))

model.add(BatchNormalization())

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=100, epochs=25,verbose=1)
predict = model.predict(test)

predict = np.argmax(predict, axis=1)

print(predict)
submission = pd.DataFrame()

submission['ImageId'] = [i for i in range(1, len(test)+1)]

submission['Label'] = predict

submission.to_csv('submission.csv', index=False)