# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading the dataset files train and test

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Displaying the first 5 rows of the DataFrame

train.head()
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# Survived for Ticket class, 1 = 1st / 2 = 2nd / 3 = rd

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
# Survived for Sex

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
# Creating new DataFrame

new_data_train = pd.get_dummies(train)

new_data_test = pd.get_dummies(test)
new_data_test.head()
new_data_train.head()
new_data_train.isnull().sum().sort_values(ascending=False).head(10)
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)

new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)
# Verifing the dataset Test

new_data_test.isnull().sum().sort_values(ascending=False).head(10)
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)
X = new_data_train.drop('Survived', axis=1)

y = new_data_train['Survived']
tree = DecisionTreeClassifier(max_depth=3, random_state=0)

tree.fit(X, y)
tree.score(X , y)