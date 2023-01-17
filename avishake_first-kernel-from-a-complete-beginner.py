# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.head(10)
train.info()
train.isnull().sum()
g = sns.catplot(x="Embarked", y="Survived", hue="Sex", data=train, kind = "bar")
train.describe()
sns.catplot(x = "Pclass", y = "Age", data = train)
import seaborn as sns2
mean_value = train.Age.mean()

train.Age.fillna(mean_value,inplace=True)

sns2.distplot(train.Age)
Common_values = 'S'

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(Common_values)
train.head()
mean_value=test.Age.mean()

test.Age.fillna(mean_value,inplace=True)
train.Sex=train.Sex.map({'male':0,'female':1})

train.Embarked=train.Embarked.map({'S':0,'C':1,'Q':2})

test.Sex=test.Sex.map({'male':0,'female':1})

test.Embarked=test.Embarked.map({'S':0,'C':1,'Q':2})
mean_value=test.Fare.mean()

test.Fare.fillna(mean_value,inplace=True)
train=train.drop(['Name','Cabin','PassengerId','Ticket'],axis=1)

test=test.drop(['Name','Cabin','PassengerId','Ticket'],axis=1)
from sklearn.model_selection import train_test_split

X = train.drop(['Survived'], axis=1)

Y = train['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100,2)

print(round(acc_log,2,), "%")