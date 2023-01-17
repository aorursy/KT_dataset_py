# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

print('Train shape:', train.shape)
train.head()

train.tail()
train.tail()
train.head()
train.isnull().sum().sort_values(ascending = False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print('Test shape:', test.shape)
print('Train columns:', train.columns.tolist())

print('Test columns:', test.columns.tolist())
sample_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

print(sample_submission.head())
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(train['Survived'])
sns.countplot(x = 'Survived', hue = 'Sex', data = train)
from sklearn.linear_model import LogisticRegression

logisticRegression = LogisticRegression()

logisticRegression.fit(X = pd.get_dummies(train['Sex']), y = train['Survived'])
test['Survived'] = logisticRegression.predict(pd.get_dummies(test['Sex']))
test[['PassengerId', 'Survived']].to_csv('kaggle_submission.csv', index = False)