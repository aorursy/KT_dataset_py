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
import sys

import numpy as np

import pandas as pd

import sklearn 

import matplotlib.pyplot as plt

import scipy as sci

import seaborn



data=pd.read_csv(r'/kaggle/input/Kannada-MNIST/train.csv')

data.shape

data=np.array(data)

X=data[:,1:785]

Y=data[:,0]

X.shape
X=X.reshape(60000, 28, 28)

X.shape
plt.subplot(221)

plt.imshow(X[0], cmap=plt.get_cmap('gray'))

plt.subplot(222)

plt.imshow(X[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)

plt.imshow(X[2], cmap=plt.get_cmap('gray'))

plt.subplot(224)

plt.imshow(X[3], cmap=plt.get_cmap('gray'))

# show the plot

plt.show()
import numpy

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K
X_train = X.reshape(X.shape[0], 28, 28,1).astype('float32')

y_train = np_utils.to_categorical(Y)
def baseline_model():

	# create model

	model = Sequential()

	model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(10, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model
model = baseline_model()

# Fit the model

model.fit(X_train, y_train,validation_split=0.2 , epochs=10, batch_size=200, verbose=2)

# Final evaluation of the model
data_1=pd.read_csv(r'/kaggle/input/Kannada-MNIST/train.csv')

data_1.shape

data_1=np.array(data_1)
X_1=data_1[:,1:785]

Y_1=data_1[:,0]
X_test = X_1.reshape(X_1.shape[0], 28, 28,1).astype('float32')
scores = model.predict(X_test)
aa=np.argmax(scores,axis=1)
submt = pd.DataFrame({'id': Y_1, 'label':aa})
submt[1:10]
filename = 'Kannda MNIST Predictions 1.csv'



submt.to_csv(filename,index=False)



print('Saved file: ' + filename)