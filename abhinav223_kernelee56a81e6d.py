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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

gs=pd.read_csv('../input/gender_submission.csv')
train.head()
train.info()
train.describe()
train['Sex']=train['Sex'].map({'male':1,'female':0})

test['Sex']=test['Sex'].map({'male':1,'female':0})
train.head()
train.corr()
train[['Age']].plot(kind='hist')
plt.scatter(train['SibSp'],train['Age'])
train[train['SibSp']==0]['Age'].median()

train[train['SibSp']==1]['Age'].median()

train[train['SibSp']==2]['Age'].median()

train[train['SibSp']==3]['Age'].mean()

train[train['SibSp']==4]['Age'].mean()

train[train['SibSp']==5]['Age'].mean()
train.loc[train.SibSp<1,'Age']=train.loc[train.SibSp<1,'Age'].fillna(29)

train.loc[train.SibSp<2,'Age']=train.loc[train.SibSp<2,'Age'].fillna(30)
train.loc[train.SibSp<3,'Age']=train.loc[train.SibSp<1,'Age'].fillna(23)
train.loc[train.SibSp<4,'Age']=train.loc[train.SibSp<4,'Age'].fillna(13.9)
train.loc[train.SibSp<5,'Age']=train.loc[train.SibSp<5,'Age'].fillna(7.05)
train.loc[train.SibSp<6,'Age']=train.loc[train.SibSp<6,'Age'].fillna(10.2)
test.loc[test.SibSp<1,'Age']=test.loc[test.SibSp<1,'Age'].fillna(29)

test.loc[test.SibSp<2,'Age']=test.loc[test.SibSp<2,'Age'].fillna(30)

test.loc[test.SibSp<3,'Age']=test.loc[test.SibSp<3,'Age'].fillna(23)

test.loc[test.SibSp<4,'Age']=test.loc[test.SibSp<4,'Age'].fillna(13.9)

test.loc[test.SibSp<5,'Age']=test.loc[test.SibSp<5,'Age'].fillna(7.05)

test.loc[test.SibSp<6,'Age']=test.loc[test.SibSp<6,'Age'].fillna(10.2)
test.fillna(25,inplace=True)
train.columns
train.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'],axis=1,inplace=True)

test.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'],axis=1,inplace=True)
train.head()
train.isnull().sum()
fig = plt.figure(figsize=(25, 8))

sns.violinplot(x='Sex', y='Age', 

               hue='Survived', data=train, 

               split=True,palette={0: "r", 1: "g"}

               

              );
plt.scatter(train['Fare'],train['Survived'])

plt.xlabel('Fare')

plt.ylabel('Survived')
train.dropna(axis=1,inplace=True)
y=train['Survived']

train.drop('Survived', axis=1, inplace=True)
from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test=tts(train,y,test_size=0.18,random_state=2)
from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB as gnb

from sklearn.tree import DecisionTreeClassifier as dtc
models = [

    RandomForestClassifier(n_estimators=100),

    MLPClassifier(),SVC(),dtc(),gnb()]
for model in models:

    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    print(score)