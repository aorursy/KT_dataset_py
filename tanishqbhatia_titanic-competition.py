import pandas as pd

import numpy as np

import matplotlib as plt

import warnings

warnings.filterwarnings('ignore')
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")

test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.head()
train_data.shape
train_data.info()
test_data.head()
test_data.shape
test_data.info()
sum(train_data['Sex']=='male') #number of males on the ship
sum(train_data['Sex']=='female') #number of females on the ship
train_data.isnull().sum() #checking the null values in the training dataset
test_data.isnull().sum() #checking the null values in the training dataset
train_data['Embarked'].fillna('S',inplace=True)

train_data.isnull().sum()
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)

train_data.isnull().sum()
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)

test_data.isnull().sum()
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

test_data.isnull().sum()
train_data.hist(color='blue',figsize=(10,10))
train_data['Age'].mean()  #The average age of people travelling in the ship
sum(train_data['Survived']==0) #the number of deaths
sum(train_data['Survived']==1) #the number of people survived
train_data['Pclass'].value_counts().plot(kind='bar')
train_data['SibSp'].value_counts().plot(kind='bar')
train_data['Parch'].value_counts().plot(kind='bar')
train_data['Survived'].value_counts().plot(kind='bar')
train_data._get_numeric_data().head()
for i in train_data['Sex']:

    if i=='male':

        train_data['Sex'].replace('male',0,inplace=True)

    else:

        train_data['Sex'].replace('female',1,inplace=True)

for i in train_data['Embarked']:

    if i=='S':

        train_data['Embarked'].replace('S',0,inplace=True)

    elif i=='C':

        train_data['Embarked'].replace('C',1,inplace=True)

    else:

        train_data['Embarked'].replace('Q',2,inplace=True)

train_data.head()        
for i in test_data['Sex']:

    if i=='male':

        test_data['Sex'].replace('male',0,inplace=True)

    else:

        test_data['Sex'].replace('female',1,inplace=True)

for i in test_data['Embarked']:

    if i=='S':

        test_data['Embarked'].replace('S',0,inplace=True)

    elif i=='C':

        test_data['Embarked'].replace('C',1,inplace=True)

    else:

        test_data['Embarked'].replace('Q',2,inplace=True)

test_data.head()        
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier
X_train=train_data.drop(['Survived','Name','Cabin','Ticket'],axis=1)

Y_train=train_data['Survived']

X_test=test_data.drop(['Name','Cabin','Ticket'],axis=1).copy()

X_train.shape,Y_train.shape,X_test.shape
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) , 3)

acc_log
X_test['Survived']=Y_pred
result=X_test[['PassengerId','Survived']]

#result.to_csv('Titanic_results.csv', sep='\t', encoding='utf-8')

result.to_csv("Titanic-results.csv", index=False)

print("The submission is completed")