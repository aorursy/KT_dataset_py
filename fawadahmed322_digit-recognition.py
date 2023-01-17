# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

print(data.shape)

targets = data.values[:, 0]
x1 = data.values[:, 1:]

data = x1.reshape((x1.shape[0], 28, 28, 1))

#data = pd.DataFrame(x1)
#targets = pd.DataFrame(y1)

print('done')
print(data.shape)
print(targets.shape)
from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, BatchNormalization, Dense, AveragePooling2D, concatenate, Lambda, Input, Flatten
from keras.regularizers import l2
from sklearn import preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf
import os
#from google.colab import files
import matplotlib.pyplot as plt
import random
from keras import backend as K
from skimage.io import imread
import cv2
from keras.utils import to_categorical

Y = to_categorical(targets)
mdl = Sequential()
mdl.add(Dense(784, input_dim = 784))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(BatchNormalization())
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dropout(0.3))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dense(500, activation = 'relu'))
mdl.add(Dropout(0.3))
mdl.add(Dense(10, activation = 'softmax'))
mdl.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.reshape(42000, 784), Y, test_size=0.25, random_state=42)
X_train = X_train/255.0
X_test = X_test/255.0
mdl.fit(X_train, y_train, epochs = 300, verbose = 1, batch_size = X_train.shape[0])
yt = np.argmax(y_test, axis = 1)

print(yt[0:10])
yp = np.argmax(mdl.predict(X_test), axis = 1)

print(np.sum(yp == yt)/y_test.shape[0])
from matplotlib import pyplot as plt
x = np.random.randint(X_test.shape[0])
#print(np.argmax(mdl.predict(X_test[x, :]),))
t = X_test[x].reshape((1, 784))
print(np.argmax(mdl.predict(t), axis = 1))
print(np.argmax(y_test[x]))
plt.imshow((X_test[x]*255).reshape(28,28))

data1 = pd.read_csv('../input/test.csv')

print(data1.shape)
predictions = mdl.predict(data1)
pd.DataFrame(predictions).to_csv('../predictions.csv')