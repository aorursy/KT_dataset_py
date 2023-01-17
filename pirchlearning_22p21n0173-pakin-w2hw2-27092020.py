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
# Read CSV
train = pd.read_csv("/kaggle/input/titanic/train.csv")

print(train.shape)

train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")

print(test.shape)

test.head()
df = pd.concat([train, test])

df.head()
df.describe()
df.isna().sum()
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(df['Title'], df['Sex'])
df['Title'] = df['Title'].replace(['Mlle','Ms','Countess','Miss','Mme', 'Dona'], 'Mrs')

df['Title'] = df['Title'].replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')

df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_dummies = pd.get_dummies(df.Title, prefix='Title')

df = pd.concat([df, title_dummies], axis=1)

df.head()
sex_dummies = pd.get_dummies(df.Sex, prefix='Sex')

df = pd.concat([df, sex_dummies], axis=1)

df.head()
embarked_dummies = pd.get_dummies(df.Embarked, prefix='Embarked')

df = pd.concat([df, embarked_dummies], axis=1)

df.head()
pclass_dummies = pd.get_dummies(df.Pclass, prefix='Pclass')

df = pd.concat([df, pclass_dummies], axis=1)

df.head()
df = df.drop(['Name', 'Sex', 'Embarked', 'Title', 'Ticket', 'Pclass', 'Cabin'], axis=1)

df.head()
df.set_index("PassengerId", inplace=True)

df.head()
df[df["Fare"].isna()]
df.loc[1044, 'Fare'] =  df[df['Pclass_3']==1]['Fare'].mean()
train_age = df[df.Age.notna()].drop('Survived', axis=1)

test_age = df[df.Age.isna()].drop('Survived', axis=1)
import xgboost as xgb

from sklearn.metrics import mean_squared_error



age_model = xgb.XGBRegressor(objective = 'reg:squarederror')

age_model.fit(train_age.drop('Age', axis=1), train_age.Age)

age_model.score(train_age.drop('Age', axis=1), train_age.Age)
test_age['Age'] = age_model.predict(test_age.drop('Age', axis=1))

test_age.head()
df.loc[df['Age'].isna(), 'Age'] = test_age['Age']

df.isna().sum()
df.describe()
df['Age'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

df['Fare'] = (df['Fare'] - df['Fare'].mean()) / df['Fare'].std()

df.head()
train = df[df['Survived'].notna()]

test = df[df['Survived'].isna()]
train.head()
test.head()