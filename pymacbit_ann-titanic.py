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
# Importing libraries

from __future__ import print_function

import pandas as pd

from keras.layers import Dense

from keras.models import Sequential





# Read the data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')





#Data Preprocessing/Cleaning

#Convert Sex of Male/Female to 1/0

train['Sex'].replace(['male', 'female'], [1, 0], inplace=True)

test['Sex'].replace(['male','female'],[1,0],inplace=True)



#Convert Embarked place C/Q/S to 0/1/2

train['Embarked'].replace(['C','Q','S'],[0,1,2], inplace=True)

test['Embarked'].replace(['C','Q','S'],[0,1,2], inplace = True)



train['Cabin'] = train['Cabin'].str.extract('(\d+)', expand=True)

test['Cabin'] = test['Cabin'].str.extract('(\d+)', expand=True)



train = train.fillna(0)

test = test.fillna(0)



y_train = train['Survived']

testName = test['PassengerId']

train.drop(['PassengerId','Survived','Name','Ticket'],inplace=True,axis=1)

test.drop(['PassengerId','Name','Ticket'],inplace=True,axis=1)



#ANN model

model = Sequential()

model.add(Dense(16, input_dim=8, init='uniform', activation='relu'))

model.add(Dense(14, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(6, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))



# Compile and Fit the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train, y_train, epochs=200, batch_size=10,  verbose=2)



# calculate predictions

predictions = model.predict(test)



# round predictions

rounded = [int(round(x[0])) for x in predictions]



#print(testName.to_frame())

output = pd.concat([testName.to_frame(),pd.DataFrame(rounded)],axis=1)

output.columns = ['PassengerId','Survived']