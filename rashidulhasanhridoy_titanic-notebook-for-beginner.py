import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import math

%matplotlib inline
data_titanic = pd.read_csv('../input/titanic/train.csv')
data_titanic.head(5)
data_titanic.columns
print('Total passengers', str(len(data_titanic)))
sns.countplot( x = 'Survived', data = data_titanic)
sns.countplot( x = 'Survived', hue = 'Sex', data = data_titanic)
sns.countplot( x = 'Survived', hue = 'Pclass', data = data_titanic)
data_titanic['Age'].hist().plot()
data_titanic['Fare'].hist().plot(bins = 20, figsize = (10, 5))
data_titanic.info()
sns.countplot( x = 'SibSp', data = data_titanic)
data_titanic.isnull()
data_titanic.isnull().sum()
sns.heatmap(data_titanic.isnull(), cmap = 'viridis')
sns.boxplot(x = 'Pclass', y = 'Age', data = data_titanic)
data_titanic.head(7)
data_titanic.drop('Cabin', axis = 1, inplace = True)
data_titanic.head(5)
data_titanic.dropna(inplace = True)
sns.heatmap(data_titanic.isnull())
data_titanic.isnull().sum()
data_titanic.head(5)
sex = pd.get_dummies(data_titanic['Sex'], drop_first = 'True')

sex.head(5)
embarked = pd.get_dummies(data_titanic['Embarked'], drop_first = True)

embarked.head(5)
Pclass = pd.get_dummies(data_titanic['Pclass'], drop_first = True)

Pclass.head(5)
data_titanic = pd.concat([data_titanic, sex, embarked, Pclass], axis = 1)

data_titanic.head(5)
data_titanic.drop(['Sex', 'Name', 'PassengerId', 'Ticket', 'Embarked'], axis = 1, inplace = True)
data_titanic.head(5)
data_titanic.drop(['Pclass'], axis = 1, inplace = True)
data_titanic.head(5)
x = data_titanic.drop('Survived', axis = 1)

y = data_titanic['Survived']
#from sklearn.cross_validation import train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)