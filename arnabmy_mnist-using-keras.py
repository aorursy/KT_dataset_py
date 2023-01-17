# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#### load dependencies
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd 
import matplotlib.pyplot as plt,matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
%matplotlib inline

# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
# Any results you write to the current directory are saved as output.

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 
y_train = train["label"]
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
X_train.shape
X_train = X_train.values.reshape(42000,784).astype('float32')

n_classes = 10
y_train = keras.utils.to_categorical(y_train,n_classes)
#train test split
X_train, X_test,y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=0)
y_test[0]
model = Sequential()
model.add(Dense((64),activation='relu',input_shape=(784,))) # 64 hidden layer, input = 784
model.add(Dense(64,activation='relu'))
model.add(Dense((10),activation='softmax')) #10 output
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128,epochs=100,verbose=1,validation_data=(X_test,y_test))
model.evaluate(X_test,y_test)
