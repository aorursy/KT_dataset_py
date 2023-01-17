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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline 



from sklearn.model_selection import train_test_split

import cv2

import keras

from matplotlib import pyplot as plt

from keras.models import Sequential 

from keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout

from keras.datasets import mnist 

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.optimizers import SGD
train = pd.read_csv('../input/digit-recognizer/train.csv')

validation = pd.read_csv('../input/digit-recognizer/test.csv')
validation.shape
train.head(10)
X = train.drop(['label'],axis = 1 )

y = train['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
fig = plt.figure(figsize = (8,8))

row, column = 5,5



for i in range(1, row * column + 1):

    im = X_train.iloc[i]

    im = np.array(im)

    fig.add_subplot(row,column, i)

    plt.imshow(im.reshape((28,28)))

plt.show()
count =train.groupby(['label']).size()

print(count)
print(X_train.shape)

print(y_test.shape)
X_train
X_train = np.array(X_train.iloc[:,:])

X_train = np.array([np.reshape(i, (28,28)) for i in X_train])



X_test = np.array(X_test.iloc[:,:])

X_test = np.array([np.reshape(i, (28,28)) for i in X_test])



num_classes = 10

y_train = np.array(y_train).reshape(-1)



y_train = np.eye(num_classes)[y_train]
print(X_train.shape)

print(X_test.shape)
X_train = X_train.reshape((33600, 28, 28, 1))

X_test = X_test.reshape((8400, 28, 28, 1))
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))

#flatten since too many dimensions, we only want a classification output

model.add(Flatten())

#fully connected to get all relevant data

model.add(Dense(128, activation='relu'))

#one more dropout for convergence' sake :) 

model.add(Dropout(0.5))

#output a softmax to squash the matrix into output probabilities

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs= 50, batch_size=32)
valid = np.array(validation.iloc[:,:])

valid = np.array([np.reshape(i, (28,28)) for i in valid])

valid = valid.reshape((28000, 28, 28, 1))
y_pred = model.predict(valid)
Y_pred = np.argmax(y_pred,axis = 1)

Y_pred
results = pd.Series(Y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit.csv",index=False)