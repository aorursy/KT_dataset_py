# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# Any results you write to the current directory are saved as output.
test=pd.read_csv(r'../input/test.csv')
train=pd.read_csv(r'../input/train.csv')
test_tmp=pd.read_csv(r'../input/test.csv')
train=train.drop(['PassengerId','Ticket'],axis=1)
train['FamilySize']=train['SibSp']+train['Parch']
train=train.drop(['SibSp','Parch'],axis=1)
train['Title']=train.Name.str.extract('([A-Za-z]+\.)',expand=False).str.replace('.','')
train=train.drop('Name',axis=1)
train=train.drop('Cabin',axis=1)
Age_Mapping=dict(train.groupby(["Title"])["Age"].mean())
train['Age1']=train.Title.map(Age_Mapping)
train.Age=train.Age.fillna(train.Age1)
train=train.drop(['Age1'],axis=1)
train.Age=train.Age.round(0)
train.Age=pd.cut(train['Age'],bins=[0,22,30,36,80],labels=[0,1,2,3],include_lowest=True)

Sex_Mapping={"male":0,"female":1}
train.Sex=train.Sex.map(Sex_Mapping)
train.Sex=train.Sex.astype(int)
train.Age=train.Age.astype(int)

train.Embarked=train.Embarked.fillna("S")
Embarked_Mapping={"S":0,"C":1,"Q":2}
train.Embarked=train.Embarked.map(Embarked_Mapping)
train.Embarked=train.Embarked.astype(int)

train.Fare=train.Fare.round(0).astype(int)
train.Fare=pd.cut(train['Fare'],bins=[0,8,14,31,512],labels=[0,1,2,3],include_lowest=True)

Title_mapping={'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':4, 'Major':4, 'Mlle':4, 'Col':4, 
                 'Don':4, 'Countess':4, 'Sir':4, 'Mme':4, 'Capt':4, 'Lady':4, 'Ms':4, 'Jonkheer':4}
train.Title=train.Title.map(Title_mapping).astype(int)
train=shuffle(train)
test=test.drop(['PassengerId','Ticket'],axis=1)
test['FamilySize']=test['SibSp']+test['Parch']
test=test.drop(['SibSp','Parch'],axis=1)
test['Title']=test.Name.str.extract('([A-Za-z]+\.)',expand=False).str.replace('.','')
test=test.drop('Name',axis=1)
test=test.drop('Cabin',axis=1)
Age_Mapping=dict(test.groupby(["Title"])["Age"].mean().fillna(0))
test['Age1']=test.Title.map(Age_Mapping)
test.Age=test.Age.fillna(test.Age1)
test=test.drop(['Age1'],axis=1)
test.Age=test.Age.round(0)
test.Age=pd.cut(test['Age'],bins=[0,22,30,36,80],labels=[0,1,2,3],include_lowest=True)

Sex_Mapping={"male":0,"female":1}
test.Sex=test.Sex.map(Sex_Mapping)
test.Sex=test.Sex.astype(int)
test.Age=test.Age.astype(int)

test.Embarked=test.Embarked.fillna("S")
Embarked_Mapping={"S":0,"C":1,"Q":2}
test.Embarked=test.Embarked.map(Embarked_Mapping)
test.Embarked=test.Embarked.astype(int)

test.Fare=test.Fare.fillna(0)
test.Fare=test.Fare.round(0).astype(int)
test.Fare=pd.cut(test['Fare'],bins=[0,8,14,31,512],labels=[0,1,2,3],include_lowest=True)

Title_mapping={'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Dr':4, 'Rev':4, 'Major':4, 'Mlle':4, 'Col':4, 
                 'Don':4, 'Countess':4, 'Sir':4, 'Mme':4, 'Capt':4, 'Lady':4, 'Ms':4, 'Jonkheer':4}

test.Title=test.Title.map(Title_mapping).fillna(4)
x=train.drop(["Survived"],axis=1)
y=train["Survived"]
(x_train,x_test,y_train,y_test)=train_test_split(x,y,test_size=0.4,random_state=200)
LogReg=LogisticRegression()
LogReg.fit(x_train,y_train)
LogReg.predict(x_test)
LogReg.score(x_test,y_test)
RClf=RandomForestClassifier()
RClf.fit(x_train,y_train)
RClf.predict(x_test)
RClf.score(x_test,y_test)
GNB=GaussianNB()
GNB.fit(x_train,y_train)
GNB.predict(x_test)
GNB.score(x_test,y_test)
pred_test=LogReg.predict(test)
submission=pd.DataFrame({"PassengerId":test_tmp['PassengerId'],"Survived":pred_test})
submission.to_csv('submission.csv', index=False)