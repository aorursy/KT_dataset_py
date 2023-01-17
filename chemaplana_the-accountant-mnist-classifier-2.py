# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

mnist = pd.read_csv('../input/train.csv')
print (mnist.info())
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.iloc[:, 1:], mnist.iloc[:, 0], 
                                                    test_size = 0.4, random_state = 0)
plt.imshow(X_train.iloc[4000, :].values.reshape(28, 28), cmap=matplotlib.cm.binary)
print (y_train.iloc[4000])
mnist_test = pd.read_csv('../input/test.csv')
print (mnist_test.info())
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
print (X_train.shape, X_test.shape)
X_train2 = X_train.astype('float32')
X_test2 = X_test.astype('float32')
X_train2 = X_train2 / 255
X_test2 = X_test2 / 255
print (y_train.shape, y_test.shape)
y_train2 = np_utils.to_categorical(y_train, 10)
y_test2 = np_utils.to_categorical(y_test, 10)
print (y_train2.shape, y_test2.shape)
model = Sequential()
model.add(Dense(784, input_shape=(784,), kernel_initializer='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10, kernel_initializer='normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train2, y_train2,
                   batch_size=200, epochs = 10,
                   verbose = 2,
                   validation_data = (X_test2, y_test2))
print (mnist_test.shape)
mnist_test2 = mnist_test / 255
yy_test = model.predict_classes(mnist_test2)
print (yy_test)
print (yy_test.shape)
mnist_submission = pd.DataFrame({'ImageId': range(1,28001), 'Label' : yy_test})
print (mnist_submission.info())
print (mnist_submission.head())
mnist_submission.to_csv('accountant_mnist3.csv', index=False)