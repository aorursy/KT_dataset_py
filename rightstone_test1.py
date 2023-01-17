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

train = pd.read_csv(r'../input/train.csv')
train.describe(include='all')
test = pd.read_csv('../input/test.csv')
train = train.drop(['Cabin'],axis=1)
test = test.drop(['Cabin'],axis=1)
train = train.drop(['Ticket'],axis = 1)
test = test.drop(['Ticket'],axis = 1)
southampton = train[train["Embarked"]=="S"].shape[0]
print("S: ",southampton)
cherbourg = train[train["Embarked"]=="C"].shape[0]
print("C: ",cherbourg)
queenstown = train[train["Embarked"]=="Q"].shape[0]
print("S: ",queenstown)
train = train.fillna({"Embarked":"S"})
embarked_mapping = {"S":1,"C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
combine = [train,test]
for dataset in combine:
    dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False)
pd.crosstab(train['Title'],train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
title_mapping = {"Mr":1,"Miss":2, "Mrs":3, "Master":4, "Royal":5, "Rare":6}
for dataset in combine:
    dataset.Title = dataset.Title.map(title_mapping)
    dataset.Title = dataset.Title.fillna(0)
    
train.head()
train = train.drop(["Name",'PassengerId'],axis =1)
test = test.drop(['Name'],axis = 1)
combine = [train,test]
train.head()
sex_mapping = {'male': 0 , 'female' :1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
train.head()
train["Age"] = train['Age'].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60,np.inf]
labels = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']
train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)
test['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)
train.head()
age_title_mapping = {1:"Young Adult", 2:"Student",3:"Adult",4:"Baby",5:"Adult",6:"Adult"}
for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x]=="Unknown":
        train["AgeGroup"][x]= age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x]=="Unknown":
        test["AgeGroup"][x]= age_title_mapping[test["Title"][x]]
age_mapping = {'Young Adult':5,'Student':4,'Adult':6,'Baby':1,"Teenager":3,'Child':2,"Senior":7}
train['AgeGroup']= train['AgeGroup'].map(age_mapping)
test['AgeGroup']= test['AgeGroup'].map(age_mapping)
train=train.drop(['Age'],axis =1)
test=test.drop(['Age'],axis =1)
train['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
test['FareBand']=pd.qcut(train['Fare'],4,labels=[1,2,3,4])
train=train.drop(['Fare'],axis =1)
test=test.drop(['Fare'],axis =1)
train.head()
train_data = train.drop('Survived', axis =1)
target = train['Survived']
train_data.info()
target.isna().sum()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10,shuffle=True, random_state=0)
k_fold
clf= RandomForestClassifier(n_estimators=13)
clf
scoring = 'accuracy'
score = cross_val_score(clf, train_data,target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#decision tree score
round(np.mean(score)*100,2)
clf.fit(train,target)
prediction = clf.predict(test)
submission = pd.DataFrame({"PassengerID": test["PassengerId"], "Survived":prediction})
submission.to_csv('submission.csv')
submission.head()