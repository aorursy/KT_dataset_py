# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the libraries

import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam, SGD

import seaborn as sns
#importing the dataset

dftrain = pd.read_csv('../input/titanic/train.csv')

dftest = pd.read_csv('../input/titanic/test.csv')
dftrain.head()
#Background

sns.scatterplot(x=dftrain.index,y=dftrain['Pclass'],hue=dftrain['Survived'],palette=["C0","C1"])
#gender bias

sns.countplot(dftrain['Sex'],hue=dftrain['Survived'])
#location vs survival

sns.scatterplot(x=dftrain.index,y=dftrain['Embarked'],hue=dftrain['Survived'])
#age vs survival

sns.scatterplot(x=dftrain.index,y=dftrain['Age'],hue=dftrain['Survived'])
#removing the unnecesary stuff

dftrain.drop(['PassengerId','Ticket','Cabin','Name'],axis=1,inplace=True)

passengerid = dftest['PassengerId']

dftest.drop(['PassengerId','Ticket','Cabin','Name'],axis=1,inplace=True)
#dummy variable encoding

sex = pd.get_dummies(dftrain['Sex'],drop_first=True)

pclass = pd.get_dummies(dftrain['Pclass'],drop_first=True)

embarked = pd.get_dummies(dftrain['Embarked'],drop_first=True)



#test set

sextest = pd.get_dummies(dftest['Sex'],drop_first=True)

pclasstest = pd.get_dummies(dftest['Pclass'],drop_first=True)

embarkedtest = pd.get_dummies(dftest['Embarked'],drop_first=True)
#dropping categorical data

dftrain.drop(['Sex','Pclass','Embarked'],axis=1,inplace=True)

#test set

dftest.drop(['Sex','Pclass','Embarked'],axis=1,inplace=True)
#concatenating dummies

dftrain = pd.concat([dftrain,sex,pclass,embarked],axis=1)

#test set

dftest = pd.concat([dftest,sextest,pclasstest,embarkedtest],axis=1)
#removing nan

dftrain.dropna(inplace=True,axis=0)

#dftest.dropna(inplace=True,axis=0)
#defining x and y array

x_train = np.array(dftrain.drop(['Survived'],axis=1))

y_train = np.array(dftrain['Survived'])

print("shape of x: ",x_train.shape)

print("shape of y: ",y_train.shape)



#test data

x_test = np.array(dftest)
#creating a model

i = Input(shape=(9))

x = Dense(10,activation='relu')(i)

x = Dense(1,activation='sigmoid')(x)

model = Model(i,x)
#compiling the model

model.compile(optimizer=Adam(learning_rate=0.0010),loss='binary_crossentropy',metrics=['accuracy'])

train = model.fit(x_train,y_train,epochs=500,shuffle=True)
#visualing the accuracy

plt.plot(train.history['accuracy'],label='train acc')

plt.legend()
#visualising the loss

plt.plot(train.history['loss'],label='train loss')

plt.legend()
#prediction

y_pred = model.predict(x_test).round()

print(y_pred)
for i in range(y_pred.shape[0]):

    print(passengerid.iloc[i],y_pred[i])