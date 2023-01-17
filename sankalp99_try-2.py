# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#print(train.info())

#print(train.head())



#print('Train :')

#print(train.isnull().sum())

#print('\nTest :')

#print(test.isnull().sum())

dataset = [train,test]



#DATA CLEANING

for i in dataset:    

    i['Age'].fillna(i['Age'].median(),inplace=True)

    i['Fare'].fillna(i['Fare'].median(),inplace=True)

    i['Embarked'].fillna(i['Embarked'].mode()[0],inplace=True)

train.drop(['Ticket','Cabin'],axis=1, inplace=True)

test.drop(['Ticket','Cabin'],axis=1, inplace=True)

#print(test.isnull().sum())



#FEATURE ENGINEERING

def extract_title(x):

        return x.split(', ')[1].split('.')[0]

def imp(x):

    if imp_title[x]<10:

        return 1

    else:

        return 0

def imp_t(x):

    if imp_title_t[x]<10:

        return 1

    else:

        return 0    

def gender(sex):

    if sex=='male':

        return 1

    elif sex=='female':

        return 0

def port(emb):

    if emb=='C':

        return 1

    elif emb=='Q':

        return 2

    elif emb=='S':

        return 3

def titles(x):

    if x=='Mr':

        return 1

    elif x=='Mrs':

        return 2

    elif x=='Miss':

        return 3

    elif x=='Master':

        return 4

    elif x=='Misc':

        return 5

for i in dataset:

    i['FamMembers'] = i['SibSp'] + i['Parch'] + 1

    i['isAlone'] = i.apply(lambda x:0 if x['FamMembers']>1 else 1,axis=1)

    i['Title'] = i.apply(lambda x:extract_title(x['Name']),axis=1)

    i['Sex'] = i.apply(lambda x : gender(x['Sex']),axis=1)

    i['Embarked'] = i.apply(lambda x:port(x['Embarked']),axis=1)

imp_title = train['Title'].value_counts().to_dict()

imp_title_t = test['Title'].value_counts().to_dict()

train['Title'] = train['Title'].apply(lambda x:'Misc' if imp(x)==1 else x)

test['Title'] = test['Title'].apply(lambda x:'Misc' if imp_t(x)==1 else x)

for i in dataset:

    i["Title"] = i.apply(lambda x: titles(x['Title']),axis=1)

train.drop('Name',axis=1,inplace=True)

test.drop('Name',axis=1,inplace=True)

train.set_index('PassengerId',inplace=True)

test.set_index('PassengerId',inplace=True)

y = train['Survived']

print(list(train))

x_f = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamMembers', 'isAlone', 'Title']

x = train[x_f]

X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=0)



clf = RandomForestClassifier()

grid_values = param_grid = { 

    'criterion' :['gini'],

    'max_depth': [100],

    'max_features': [3],

    'min_samples_leaf': [5],

    'min_samples_split': [12],

    'n_estimators': [100]

}

grid_clf = GridSearchCV(clf, param_grid = grid_values)

grid_clf.fit(X_train,y_train)

grid_clf.score(X_test,y_test)
final_test = test.copy()

final_survived = grid_clf.predict(final_test)

answer = pd.DataFrame({'PassengerID':test.index,'Survived':final_survived})

answer.to_csv('try_2_titanic.csv',index=False)


