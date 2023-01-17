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
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data_train = pd.read_csv('../input/train.csv')

x_test = pd.read_csv('../input/test.csv')
data_train.describe()
x_test.describe()
y_train = data_train["label"]

x_train = data_train.drop(labels = ["label"],axis = 1)
sns.countplot(y_train)

y_train.value_counts()

#plt.hist(y_train)
X_train = np.array(x_train[['pixel' + str(i) for i in range(0,784)]])

Y_train = np.array(y_train)

X_test = np.array(x_test[['pixel' + str(i) for i in range(0,784)]])
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
plt.imshow(X_train[3][:,:,0])
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train)
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(28,28,1)))

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
model.fit(X_train, Y_train, batch_size=100, epochs=20,verbose=1)
y = model.predict(X_test)

y = np.argmax(y, axis=1)
submission = pd.DataFrame()

submission['ImageId'] = [i for i in range(1, len(x_test)+1)]

submission['Label'] = y

submission.to_csv('submission.csv', index=False)