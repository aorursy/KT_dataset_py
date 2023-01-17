import numpy as np

from sklearn.cross_validation import train_test_split

import pandas as pd

from sklearn import preprocessing
#Loading the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head(5)
del train['Name']

del train['Ticket']

del train['Cabin']

del train['PassengerId']
del test['Name']

del test['Ticket']

del test['Cabin']

del test['PassengerId']
list(train)
survive = train['Survived']

train.drop(labels=['Survived'], axis=1,inplace = True)

train.insert(7, 'Survived', survive)
train.isnull().any()
#replaces NaN in embarked

train["Embarked"] = train["Embarked"].fillna("N")

test["Fare"] = test["Fare"].fillna("N")
#Average Age

av_age = train["Age"]

av_age = av_age.mean()

#Age correction - 8 years

av_age = av_age-8

print(av_age)
#Replace NaN with the average Age

train["Age"] = train["Age"].fillna(av_age)

test["Age"] = test["Age"].fillna(av_age)

test=test.fillna(0)
#maps sex

train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})

test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})



#maps embarked

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1,'Q': 2, 'N':3})

test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1,'Q': 2, 'N':3})
train.head(10)
test.head(10)
from ipykernel import kernelapp as app



test.columns = range(test.shape[1])

train.columns = range(train.shape[1])

test_norm=test

train_norm=train

test_norm = test.convert_objects(convert_numeric=True)

train_norm = train.convert_objects(convert_numeric=True)
#pandas to numpy

trainnum = train_norm.as_matrix([0,1,2,3,4,5,6])

testnum = test_norm.as_matrix([0,1,2,3,4,5,6])



labels = train_norm.as_matrix([7])

#make printing numpy pretty

np.set_printoptions(precision=3)
min_max_scaler = preprocessing.MinMaxScaler()

trainnum_norm = min_max_scaler.fit_transform(trainnum)
trainnum_norm [3]
testnum = np.nan_to_num(testnum)
min_max_scaler2 = preprocessing.MinMaxScaler()

testnum_norm = min_max_scaler2.fit_transform(testnum)
testnum_norm[3]
import tensorflow as tf

import keras    
from keras.models import Sequential
np.broadcast(trainnum_norm).shape
from keras.layers import Dense, Activation, Dropout



model = Sequential()

model.add(Dense(units=6240, input_dim=7))

model.add(Activation('relu'))

model.add(Dropout(0.5))



model.add(Dense(units=3000))

model.add(Activation('tanh'))

model.add(Dropout(0.5))



model.add(Dense(units=128))

model.add(Activation('tanh'))

model.add(Dropout(0.5))



model.add(Dense(units=1))

model.add(Activation('sigmoid'))
from keras import optimizers



opt=keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)



model.compile(loss='binary_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])
model.fit(trainnum_norm, labels, epochs=25, batch_size=25, validation_split=0.1)



scores = model.evaluate(trainnum_norm, labels)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
survival = model.predict(testnum_norm, verbose=1)


survived = (survival + 0.5 ).astype("int")

ids = np.asarray(list(range(892,1310)))



survive = survived.reshape(418) 



output = pd.DataFrame({ 'PassengerId' : ids,  'Survived': survive }, index =(range(891,1309)) )
output.head(8)
output.to_csv('../working/submission.csv', index=False)