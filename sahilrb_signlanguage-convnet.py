# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train.csv')

test=pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')

train.head()
print(train.shape)

print(test.shape)
X_train = train.drop('label', axis = 1)

X_test = test.drop('label', axis = 1)

Y_train = train['label']

Y_test = test['label']
X_train.head()
Y_train.head()
plt.figure(figsize = (12,7))

sns.countplot(x = 'label', data = train)
images_train = X_train.values

images_train = images_train.reshape(-1, 28,28,1)

images_train.shape
images_test = X_test.values

images_test = images_test.reshape(-1,28,28,1)

images_test.shape
images_train.mean()
images_train = images_train / 255

images_test = images_test / 255
images_train.mean()
fig,axe=plt.subplots(2,2)

fig.suptitle('Preview of dataset')

axe[0,0].imshow(images_train[0].reshape(28,28),cmap='gray')

axe[0,0].set_title('label: 3  letter: C')

axe[0,1].imshow(images_train[1].reshape(28,28),cmap='gray')

axe[0,1].set_title('label: 6  letter: F')

axe[1,0].imshow(images_train[2].reshape(28,28),cmap='gray')

axe[1,0].set_title('label: 2  letter: B')

axe[1,1].imshow(images_train[4].reshape(28,28),cmap='gray')

axe[1,1].set_title('label: 13  letter: M')
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

Y_train = lb.fit_transform(Y_train)

Y_test = lb.fit_transform(Y_test)
print(Y_train.shape)

print(Y_test.shape)
from sklearn.model_selection import train_test_split
images_train, images_val, Y_train, Y_val = train_test_split(images_train, Y_train, test_size = 0.07, random_state = 5)

print(images_train.shape)

print(images_val.shape)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import ReLU
conv_model = Sequential()

conv_model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', input_shape = (28,28,1), activation = 'relu'))

conv_model.add(MaxPooling2D(pool_size = (2,2)))

conv_model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation='relu'))

conv_model.add(MaxPooling2D(pool_size = (2,2)))

conv_model.add(Flatten())

conv_model.add(Dense(units = 256, activation='relu', kernel_regularizer='l2'))

conv_model.add(Dropout(0.2))

conv_model.add(Dense(units = 24, activation='softmax'))





conv_model.summary()
conv_model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
conv_model.fit(images_train, Y_train, epochs = 10, validation_data = (images_val, Y_val), verbose = 2)
ls,acc = conv_model.evaluate(images_test, Y_test)
print('Test set loss:'+str(ls)[0:6])

print('Test set accuracy:'+str(acc)[0:6])
conv_deep_model = Sequential()

conv_deep_model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same'))

conv_deep_model.add(BatchNormalization())

conv_deep_model.add(ReLU())

conv_deep_model.add(MaxPooling2D(pool_size = (2,2)))



conv_deep_model.add(Conv2D(filters = 64, kernel_size = (5,5), padding = 'same'))

conv_deep_model.add(BatchNormalization())

conv_deep_model.add(ReLU())

conv_deep_model.add(MaxPooling2D(pool_size = (2,2)))



conv_deep_model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same'))

conv_deep_model.add(BatchNormalization())

conv_deep_model.add(ReLU())

conv_deep_model.add(AveragePooling2D(pool_size = (2,2)))



conv_deep_model.add(Flatten())

conv_deep_model.add(Dense(units = 256, activation = 'relu', kernel_regularizer = 'l2'))

conv_deep_model.add(Dropout(0.2))

conv_deep_model.add(Dense(units = 24, activation = 'softmax'))
conv_deep_model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
conv_deep_model.fit(images_train, Y_train, epochs = 15, validation_data = (images_val, Y_val), verbose = 2)
ls,acc = conv_deep_model.evaluate(images_test, Y_test)
print('Test set loss:'+str(ls)[0:6])

print('Test set accuracy:'+str(acc)[0:6])