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
#Train

#Test



#img = Train.iloc[2,1:]



img = np.array(Train.iloc[2,1:])    #numpy for 28*28; numpy sara img reshape kora jay na

img = img.reshape(28,28)

img.shape
import matplotlib.pyplot as plt

plt.imshow(img)

plt.show()
X_train = np.array(Train.iloc[:,1:])

y_train = np.array(Train.iloc[:,0])
X_train.shape,y_train.shape
X_test = np.array(Test.iloc[:,1:])

y_test = np.array(Test.iloc[:,0])
X_test.shape,y_test.shape
import keras
#for creating one hot encoding



from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,10)

y_test = np_utils.to_categorical(y_test,10)
y_train.shape, y_test.shape
y_train[0]
#For creating layer



from keras.models import Sequential

from keras.layers.core import Dense, Activation 



model = Sequential()  #obj call

model.add(Dense(128,input_shape=(784,)))   #hidden layer 128 neuron

model.add(Activation('relu'))             #relu dara normalize kora hoy; 128 ta neuron er normalize value

model.add(Dense(10))                      #output layer 10 neuron

model.add(Activation('softmax'))          #for multi-class; for binary-class --> sigmoid (thershold value dara vag kore dey)
model.summary()
model.compile(loss = "categorical_crossentropy",metrics=["accuracy"],optimizer = "adam")
model.fit(X_train,y_train)
model.fit(X_train,y_train,epochs= 100)