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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data  = pd.read_csv('../input/train.csv')

display(data.head())

display(data.describe())

display(data.info())
ytrain = data['label']

xtrain = data.drop('label',axis =1)



ytrain.describe()
xtrain /=255.0

xtrain.describe()
from keras.utils.np_utils import to_categorical

ytraincat = to_categorical(ytrain)
from keras.models import Sequential

from keras.layers import Dense

import keras.backend as kb

kb.clear_session()



model = Sequential()

model.add(Dense(512, input_dim = 28*28 , activation= 'relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(32 , activation = 'relu'))

model.add(Dense(10 , activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'rmsprop' , metrics = ['accuracy'])
h = model.fit(xtrain, ytraincat, batch_size=128, epochs=10, verbose=1, validation_split=0.3)
xtest = pd.read_csv('../input/test.csv')
display(xtest.describe())

xtest/=255.0
results = model.predict(xtest)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("../kaus_mnist_submission.csv",index=False)