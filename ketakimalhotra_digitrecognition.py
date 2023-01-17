# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.utils import to_categorical


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from keras.datasets import mnist
(trainx, trainy), (testx, testy) = mnist.load_data()

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(trainx[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(trainy[i]))
  plt.xticks([])
  plt.yticks([])
print(trainx.shape)
print(testx.shape)
X_train = trainx.reshape(trainx.shape[0], 28, 28, 1)
X_test = testx.reshape(testx.shape[0],28, 28,1 )
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train.shape
#one-hot encode target column
num_category = 10
y_train = to_categorical(trainy, num_category)
y_test = to_categorical(testy, num_category)
from digitrecognition_script import train
model = train(X_train , y_train)
score= model.evaluate(X_test, y_test, verbose= 0)
    

accuracy= round(score[1] * 100, 2)
accuracy
for xt in X_test:
    #printing predictions on test-set
    print(np.argmax(model.predict(xt.reshape(1,28,28,1))))