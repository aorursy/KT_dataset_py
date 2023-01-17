# Imports

import pandas as pd



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
# get titanic & test csv files as a DataFrame

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# preview the data

train.head()
train.loc[:,'Title'] = train.loc[:,'Name'].apply(lambda x : x.split(', ')[1].split('.')[0])

print(train['Title'].value_counts())



train.loc[train['Title']=='Mlle','Title'] = 'Miss'

train.loc[train['Title']=='Ms','Title'] = 'Miss'

train.loc[train['Title']=='Mme','Title'] = 'Mrs'



train.loc[~train['Title'].isin(['Mr','Mrs','Master','Miss']),'Title'] = 'Hon'

print(train['Title'].value_counts())
train['FSize'] = train['Parch'] + train['SibSp'] + 1
# Embarked

print('Look at ticket info for the rows missing "Embarked" entry:')

print(train.loc[train['Embarked'].isnull(),['Pclass','Fare','Embarked']])



print('\nFind median Fare for 1st class passengers with each embarkment:')

for e in ['C','S','Q']:

    print(e, train.loc[(train['Pclass']==1) & (train['Embarked']==e),'Fare'].median())



print('\nSo we\'ll set the missing values to "C".')

train.loc[[61,829],'Embarked'] = 'C'
# Fare

print('Missing', sum(test['Fare'].isnull()), 'test passenger fare(s).')

print('\nLook at ticket info for the rows missing "Fare" entry:')

print(test.loc[test['Fare'].isnull(),['Pclass','Fare','Embarked']])



print('\nFind median Fare for 3rd class passengers with embarkment:')

print(train.loc[(train['Pclass']==3) & (train['Embarked']=='S'),'Fare'].median())



print('\nSo we\'ll set the missing value to 8.05.')

test.loc[152,'Fare'] = 8.05
# Age 

from fancyimpute import SimpleFill

Xtmp = train.drop(['PassengerId','Name','Sex','Title','Ticket','Cabin','Embarked'],axis=1)

Xtmp = pd.DataFrame(SimpleFill().complete(Xtmp),index=Xtmp.index,columns=Xtmp.columns)

train.loc[:,'Age'] = Xtmp.loc[:,'Age']
train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
test.loc[:,'Title'] = test.loc[:,'Name'].apply(lambda x : x.split(', ')[1].split('.')[0])

print(test['Title'].value_counts())



test.loc[test['Title']=='Mlle','Title'] = 'Miss'

test.loc[test['Title']=='Ms','Title'] = 'Miss'

test.loc[test['Title']=='Mme','Title'] = 'Mrs'



test.loc[~test['Title'].isin(['Mr','Mrs','Master','Miss']),'Title'] = 'Hon'

print(test['Title'].value_counts())



test['FSize'] = test['Parch'] + test['SibSp'] + 1



Xtmp = test.drop(['PassengerId','Name','Sex','Title','Ticket','Cabin','Embarked'],axis=1)

Xtmp = pd.DataFrame(SimpleFill().complete(Xtmp),index=Xtmp.index,columns=Xtmp.columns)

test.loc[:,'Age'] = test.loc[:,'Age']

pid = test['PassengerId'].copy()



test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)



test['Sex'] = test['Sex'].map({'male':1,'female':0})

test['Embarked'] = test['Embarked'].map({'S':0,'C':1,'Q':2})

test['Title'] = test['Title'].map({'Mr':0,'Miss':1,'Master':2,'Mrs':3,'Hon':4})



print(test.head())
train['Sex'] = train['Sex'].map({'male':1,'female':0})

train['Embarked'] = train['Embarked'].map({'S':0,'C':1,'Q':2})

train['Title'] = train['Title'].map({'Mr':0,'Miss':1,'Master':2,'Mrs':3,'Hon':4})
from sklearn.model_selection import train_test_split



y = train['Survived'].copy()

X = train.drop('Survived',axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Random Forests



rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)

print(sum(y_pred != y_test))
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X, y)



test['Age'] = test['Age'].fillna(29.7)



y_pred = rf.predict(test)
submission = pd.DataFrame({

        "PassengerId": pid,

        "Survived": y_pred

    })

submission.to_csv('my_titanic.csv', index=False)