import numpy as np

import pandas as pd
trainData = pd.read_csv('../input/train.csv')

testData = pd.read_csv("../input/test.csv")
print(trainData.dtypes.sort_values())

print(testData.dtypes.sort_values())
trainData.isnull().sum()[trainData.isnull().sum()>0]
testData.isnull().sum()[testData.isnull().sum()>0]
trainData.Age=trainData.Age.fillna(trainData.Age.mean())

testData.Age=testData.Age.fillna(trainData.Age.mean())



trainData.Fare=trainData.Fare.fillna(trainData.Fare.mean())

testData.Fare=testData.Fare.fillna(trainData.Fare.mean())





trainData.Embarked=trainData.Embarked.fillna(trainData.Embarked.mode()[0])

testData.Embarked=testData.Embarked.fillna(trainData.Embarked.mode()[0])
trainData.head()

testData.head()

trainData.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)

testData.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)
testData.head()

combined=pd.concat([trainData, testData], sort=False)

print(combined.dtypes.sort_values())
length = trainData.shape[0]

combined=pd.concat([trainData, testData], sort=False)

combined=pd.get_dummies(combined)

trainData=combined[:length]

testData=combined[length:]



trainData.Survived=trainData.Survived.astype('int')
x=trainData.drop("Survived",axis=1)

y=trainData['Survived']

xtest=testData.drop("Survived",axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error
RF = RandomForestClassifier(random_state=1)

results = cross_val_score(RF,x,y,scoring='accuracy',cv=5)

print(results)

np.mean(results)

RF.fit(x, y)

print(RF)
predictions=RF.predict(xtest)

column_name = pd.read_csv('../input/test.csv')

output=pd.DataFrame({'PassengerId':column_name['PassengerId'],'Survived':predictions})

output.to_csv('submission.csv', index=False)