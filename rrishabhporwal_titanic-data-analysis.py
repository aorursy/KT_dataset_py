# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/titanic_data.csv")
dataset.head()
print("Number of passenger in original data is : "+str(len(dataset['PassengerId'])))
sns.countplot(x='Survived', data=dataset)
sns.countplot(x="Survived", hue="Sex", data=dataset)
sns.countplot(x="Survived", hue="Pclass", data=dataset)
dataset['Age'].plot.hist()
dataset['Fare'].plot.hist(bins=20, figsize=(10,5))
dataset.info()
sns.countplot(x="SibSp", data=dataset)
dataset.isnull()
dataset.isnull().sum()
sns.heatmap(dataset.isnull(),cmap="viridis")
dataset.drop('Cabin', axis=1, inplace=True)

dataset.head()
sns.heatmap(dataset.isnull())
dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset.isnull().sum()
sns.heatmap(dataset.isnull())
dataset.head(3)
sex = pd.get_dummies(dataset['Sex'], drop_first=True)

sex.head()
embarked = pd.get_dummies(dataset['Embarked'], drop_first=True)

embarked.head(5)
pcl = pd.get_dummies(dataset['Pclass'], drop_first=True)

pcl.head()
dataset = pd.concat([dataset, sex, embarked, pcl], axis=1)
dataset.head()
dataset.drop(['Sex','Embarked','PassengerId','Name','Ticket'], axis=1, inplace=True)
dataset
dataset.drop(['Pclass'], axis=1, inplace=True)
dataset.head(5)
X = dataset.drop('Survived', axis=1)

y = dataset['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test, prediction)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, prediction)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)