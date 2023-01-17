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
import pandas as pd

Train = pd.read_csv("../input/mnist_train.csv")

Test = pd.read_csv("../input/mnist_test.csv")
Train.head()

Test
img = np.array(Train.iloc[6][1:])
img.shape

img = img.reshape(28,28)

img.shape
import matplotlib.pyplot as plt

plt.imshow(img)

plt.show()
X_train = np.array(Train.iloc[:,1:])

Y_train = np.array(Train.iloc[:,0])
X_train.shape,Y_train.shape
X_test = np.array(Test.iloc[:,1:])

Y_test = np.array(Test.iloc[:,0])
X_test.shape,Y_test.shape
import keras
from keras.utils import np_utils 

Y_train = np_utils.to_categorical(Y_train, 10)

Y_test = np_utils.to_categorical(Y_test, 10)
Y_train.shape, Y_test.shape
Y_test[0]
from keras.models import Sequential

from keras.layers.core import Dense, Activation 

model = Sequential()

model.add(Dense(512,input_shape=(784,)))

model.add(Activation('relu'))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Activation('softmax')) # fror binary classifier sigmoid
model.summary()

#128*10+ 10-<weight to calculate param
model.compile(loss="categorical_crossentropy", metrics =["accuracy"] , optimizer ="adam")  #for binary classification "binary_crossentropy"
model.fit(X_train, Y_train, epochs = 100)