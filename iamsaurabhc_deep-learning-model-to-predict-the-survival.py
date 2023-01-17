from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from __future__ import print_function



import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import RMSprop



# Read the data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/gender_submission.csv')



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



#print (train.iloc[0])

print(testName.head)

# create model

model = Sequential()

model.add(Dense(16, input_dim=8, init='uniform', activation='relu'))

model.add(Dense(12, init='uniform', activation='relu'))

model.add(Dense(8, init='uniform', activation='relu'))

model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model

model.fit(train, y_train, epochs=150, batch_size=10,  verbose=2)

# calculate predictions

predictions = model.predict(test)

# round predictions

rounded = [int(round(x[0])) for x in predictions]

#print(testName.to_frame())

output = pd.concat([testName.to_frame(),pd.DataFrame(rounded)],axis=1)

output.columns = ['PassengerId','Survived']

print(output)

output.to_csv('submission.csv',index=False)