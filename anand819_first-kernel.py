# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')
print(train_df.head())

print(test_df.head())
# Remove Passenger Id from both df's

train_df.drop('PassengerId', axis=1, inplace=True)

test_id = test_df['PassengerId']

test_df.drop('PassengerId', axis=1, inplace=True)
# Missing Values

train_na_df = train_df.isna().sum()/ train_df.shape[0]

print("Missing values for training data")

print(train_na_df[train_na_df>0])

test_na_df = test_df.isna().sum()/ test_df.shape[0]

print("Missing values for testing data")

print(test_na_df[test_na_df>0])
# Target distribution 

sns.countplot(train_df['Survived'])
sns.countplot(x='Pclass', hue='Survived', data=train_df)
sns.countplot(x='Sex', hue='Survived', data=train_df)
fig = plt.figure(figsize=(10, 4))

fig.add_subplot(1, 3, 1)

sns.distplot(train_df['Age'].fillna(train_df['Age'].median()))

fig.add_subplot(1, 3, 2)

sns.boxplot(y='Age', x='Survived', data=train_df)
fig = plt.figure(figsize=(10, 4))

fig.add_subplot(1, 2, 1)

sns.distplot(train_df['Fare'].fillna(train_df['Fare'].median()))

fig.add_subplot(1, 2, 2)

sns.boxplot(y='Fare', x='Survived', data=train_df)
sns.countplot(x='Embarked', hue='Survived', data=train_df)