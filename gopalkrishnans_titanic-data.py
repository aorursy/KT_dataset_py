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
import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns

import matplotlib.pyplot as plt


train = pd.read_csv("../input/train.csv")



test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.describe()
test.describe()
print(train.columns.values)
print(train.isna().sum())
print(test.isna().sum())
train.fillna(train.mean(),inplace=True)
test.fillna(test.mean(),inplace=True)
print(train.isna().sum())
print(test.isna().sum())
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean()
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
train.info()
train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
labelEncoder=LabelEncoder()

labelEncoder.fit(train['Sex'])

labelEncoder.fit(test['Sex'])

train['Sex'] = labelEncoder.transform(train['Sex'])

test['Sex'] = labelEncoder.transform(test['Sex'])
print(train.Sex.head())
print(test.Sex.head())
train.info()
test.info()
test.info()
train_X = train[['PassengerId','Sex','Pclass','Age','SibSp','Parch','Fare']]
train_Y = train['Survived']

test_X = test[['PassengerId','Sex','Pclass','Age','SibSp','Parch','Fare']]
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

neigh = KNeighborsClassifier(n_neighbors=2)



#Train the algorithm

neigh.fit(train_X, train_Y)



# predict the response

pred = neigh.predict(test_X)

target = train.Survived

print(pred)

# evaluate accuracy

#print ("KNeighbors accuracy score : ",accuracy_score(pred,))