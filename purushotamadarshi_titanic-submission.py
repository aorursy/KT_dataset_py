# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import random



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')
train_data.columns
train_data = train_data.drop(['PassengerId','Ticket','Cabin'], axis=1)

test_data = test_data.drop(['PassengerId','Ticket','Cabin'], axis=1)
train_data['title'] = train_data['Name'].str.extract('([A-Za-z]+)\.')
test_data['title'] = test_data['Name'].str.extract('([A-Za-z]+)\.')
train_data.sample()
test_data.sample()
train_data['title'] = train_data['title'].replace(['Dr','Rev','Col','Major','Don','Capt','Jonkheer','Sir','Master'],'Mr')

train_data['title'] = train_data['title'].replace(['Mlle','Lady','Countess','Ms','Mme'],'Mrs')



test_data['title'] = test_data['title'].replace(['Dr','Rev','Col','Major','Don','Capt','Jonkheer','Sir','Master'],'Mr')

test_data['title'] = test_data['title'].replace(['Mlle','Lady','Countess','Ms','Mme'],'Mrs')
train_data['Title'] = train_data['title'].map({'Mr':0,'Mrs':1,'Miss':2})

test_data['Title'] = test_data['title'].map({'Mr':0,'Mrs':1,'Miss':2})
train_data['Title'].value_counts()
train_data.describe(include=['O'])
train_data['Embarked'].value_counts()
train_data['Embarked'] = train_data['Embarked'].map({'S':0,'C':1,'Q':2})
test_data['Embarked'] = test_data['Embarked'].map({'S':0,'C':1,'Q':2})
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
train_data['Embarked'].fillna(0, inplace=True)
test_data['Embarked'].fillna(0, inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
train_data.isna().sum()
# just one nan value so we push him into male figure`

test_data['Title'].fillna(0, inplace=True)
# if the person is alone we give him 0 and if with family then we give him 1

train_data['alone'] = 0

train_data.loc[train_data['SibSp']+train_data['Parch'] > 0, 'alone'] =1
# if the person is alone we give him 0 and if with family then we give him 1

test_data['alone'] = 0

test_data.loc[test_data['SibSp']+test_data['Parch'] > 0, 'alone'] =1
train_data['Sex'] = train_data['Sex'].map({'male':0,'female':1})

test_data['Sex'] = test_data['Sex'].map({'male':0,'female':1})
train_data.drop(['Name','SibSp','Parch','title'],axis=1, inplace=True)

test_data.drop(['Name','SibSp','Parch','title'],axis=1, inplace=True)
train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0

train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1

train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2

train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3

train_data.loc[ train_data['Age'] > 64, 'Age'] = 4



test_data.loc[ test_data['Age'] <= 16, 'Age'] = 0

test_data.loc[(test_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1

test_data.loc[(test_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2

test_data.loc[(test_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3

test_data.loc[ test_data['Age'] > 64, 'Age'] = 4
train_data.head()
test_data.head()
xtrain = train_data.iloc[:,1:].values

xtest = test_data.iloc[:,:].values



ytrain = train_data.iloc[:,0]
# trying Logistic regression

lr = LogisticRegression()

lr.fit(xtrain, ytrain)

ac_score = lr.score(xtrain,ytrain)

ac_score
print(lr.coef_)
# prediction for the logistic regression

yp_lr = lr.predict(xtest)
# trying SVC



svc= SVC()

svc.fit(xtrain, ytrain)

ac_score = svc.score(xtrain, ytrain)

ac_score
yp_svc = svc.predict(xtest)
# trying knn

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(xtrain, ytrain)

knn.score(xtrain, ytrain)
yp_knn= knn.predict(xtest)
# trying Naive Bayes



nb = GaussianNB()

nb.fit(xtrain, ytrain)

nb.score(xtrain, ytrain)
yp_nb = nb.predict(xtest)
# trying rabndom forest

rf = RandomForestClassifier(n_estimators=100)

rf.fit(xtrain, ytrain)

rf.score(xtrain, ytrain)
yp_rf = rf.predict(xtest)
yp_rf
test_data
sub_test_data = pd.read_csv('../input/titanic/test.csv')
sub_test_data
submission = pd.DataFrame({"PassengerId": sub_test_data["PassengerId"],"Survived": yp_rf})
submission.to_csv("../titanic_submission.csv")