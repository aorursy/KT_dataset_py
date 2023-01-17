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
import pandas as pd
x=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_train.csv")
y=pd.read_csv("/kaggle/input/mnist-in-csv/mnist_test.csv")
y1=y.iloc[:,1:].values
ytrain=x.iloc[:,0].values
xtrain=x.iloc[:,1:].values
xtrain.shape
ytrain=ytrain.reshape(-1,1)
len(y1[0])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
o=OneHotEncoder()
ytrain=o.fit_transform(ytrain).toarray()
len(ytrain[0])
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
model = Sequential()
model.add(Dense(392, input_shape=(784,)))
model.add(Activation('relu'))                            
model.add(Dense(392))
model.add(Activation('relu'))
model.add(Dense(392))
model.add(Activation('relu'))
model.add(Dense(392))
model.add(Activation('relu'))
model.add(Dense(392))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrain,ytrain,batch_size=10000,nb_epoch=50)
ypred=model.predict(y1)
ypred=o.inverse_transform(ypred)
ypred
