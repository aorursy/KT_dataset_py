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
trainData = pd.read_csv('../input/fashion-mnist_train.csv')

testData = pd.read_csv('../input/fashion-mnist_test.csv')
xTrain = trainData.iloc[:, 1:].values

yTrain = trainData.iloc[:, 0].values 

xTest = testData.iloc[:, 1:].values

yTest = testData.iloc[:, 0].values
from keras.utils import np_utils
#Reshaping for CNN

xTrain = xTrain.reshape((xTrain.shape[0], 28, 28, 1))

xTest = xTest.reshape((xTest.shape[0], 28, 28, 1))

yTrain = np_utils.to_categorical(yTrain)

yTest = np_utils.to_categorical(yTest)
import matplotlib.pyplot as plt
plt.imshow(xTrain[5].reshape(28, 28), cmap = 'gray')

plt.show()
from keras.models import Sequential

from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()

model.add(Convolution2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)))

model.add(Convolution2D(64, (3,3), activation = 'relu'))

model.add(Dropout(0.25))

model.add(MaxPooling2D(2,2))

model.add(Convolution2D(32, (5,5), activation = 'relu'))

model.add(Convolution2D(8, (5,5), activation = 'relu'))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(xTrain, yTrain, epochs = 75, shuffle = True, batch_size = 4096)
plt.figure(0)

plt.style.use('seaborn')

plt.plot(hist.history['loss'], 'blue')

plt.plot(hist.history['acc'], 'green')

plt.legend(loc='upper right')

plt.show()
score = model.evaluate(xTest,yTest,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(score[1]))