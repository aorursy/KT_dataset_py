# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# data analysis

import pandas as pd

import numpy as np

import random as rnd



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



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#acquiring datasets, defining, and printing overview

train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

combine = [train_df, test_df]

train_data.head()
#print more deets

print(train_df.shape)

print(train_df.columns.values)
#stastical overview

train_df.describe()
#find null values

train_df.tail() 
#make all values numbers - sex

for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
#make all values numbers - SibSp + Parch = FamilySize

for dataset in combine:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Embarked numberfication pt1 - finding the most frequent port

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
#Embarked numberfication pt2 - replacing null Embarked values with freq_port

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Embarked numberfication pt2 - numberfying values

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
# Title info

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
#combining like titles

for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Don', 'Sir', 'Jonkheer'], 'Royalty')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr', 'Major', 'Rev'], 'Officer')

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr', 'Major', 'Rev'], 'Officer') 

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#Fare simplification pt 1 - replace null fare values w median value

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
# Finding highest correlation columns w Logistic Regression pt 1 - defining values



X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Finding highest correlation columns w Logistic Regression pt 2 - setting up LogReg



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log