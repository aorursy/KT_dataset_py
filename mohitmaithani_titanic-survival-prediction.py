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
data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')

# data_train.head()

data_train.tail(20)
data_train.describe().T
data_train.info()
sns.set_style('darkgrid')

sns.countplot(data=data_train, x='Survived', hue='Sex')
sns.countplot(data=data_train, x='Survived',hue='Pclass')
sns.distplot(data_train['Age'].dropna(), bins=30)
figure = plt.figure(figsize=(10,6))

sns.boxplot(data=data_train, x='Pclass', y='Age')
first_mean = round(data_train[data_train['Pclass'] == 1]['Age'].dropna().mean())

second_mean = round(data_train[data_train['Pclass'] == 2]['Age'].dropna().mean())

third_mean = round(data_train[data_train['Pclass'] == 3]['Age'].dropna().mean())



# creating function to fill missing age

def filling(col):

    Age = col[0]

    Pclass = col[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return first_mean

        elif Pclass == 2:

            return second_mean

        else:

            return third_mean

    else:

        return Age



data_train['Age'] = data_train[['Age', 'Pclass']].apply(filling, axis=1)
sex = pd.get_dummies(data_train['Sex'],drop_first=True)

embarked = pd.get_dummies(data_train['Embarked'],drop_first=True)
X_train = pd.concat([data_train[['Age', 'SibSp', 'Parch']], sex, embarked], axis=1)

y_train = data_train['Survived']
data_test['Age'] = data_test[['Age', 'Pclass']].apply(filling, axis=1)

sex = pd.get_dummies(data_test['Sex'],drop_first=True)

embarked = pd.get_dummies(data_test['Embarked'],drop_first=True)

X_test = pd.concat([data_test[['Age', 'SibSp', 'Parch']], sex, embarked], axis=1)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

model.score(X_train, y_train)
predictions = model.predict(X_test)

data_test['Survived'] = predictions
submit = data_test[['PassengerId', 'Survived']]

submit.to_csv('submission.csv', index=False)
y_test=data_test['Survived']
# from sklearn.model_selection import train_test_split

# X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

print(X_train,X_test,y_train,y_test)