# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import warnings;

warnings.filterwarnings("ignore");

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

test_sur = pd.read_csv("../input/gender_submission.csv")

test = pd.read_csv("../input/test.csv")



data['Age'][np.isnan(data['Age'])] =  np.nanmedian(data['Age'])

data = data.dropna()

test = test.drop(columns = ['Name','Ticket','Cabin','PassengerId'])

test['Age'][np.isnan(test['Age'])] = np.nanmedian(test['Age'])

test['Fare'][np.isnan(test['Fare'])] = np.nanmedian(test['Fare'])

test_sur = test_sur.drop(columns = ['PassengerId'])
data = data.drop(columns = ['Name','PassengerId','Ticket','Cabin'])

map1 = {"female":0 , "male":1}

map2 = {"S":0, "C":1, "Q":2}

data['Sex'] = data.Sex.map(map1)

data['Embarked'] = data.Embarked.map(map2)

data = pd.get_dummies(data)

test = pd.get_dummies(test)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,1:],data.iloc[:,0],test_size=0.3)
model = RandomForestClassifier()

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print(accuracy_score(y_test,predictions))
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Dropout,Activation

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train,2)

y_test = np_utils.to_categorical(y_test,2)

def clf_model():

    model = Sequential()

    model.add(Dense(7,input_dim=7,kernel_initializer='normal',activation='relu'))

    model.add(Dense(32,activation='relu'))

    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])

    return model
model = clf_model()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=5,verbose=1)