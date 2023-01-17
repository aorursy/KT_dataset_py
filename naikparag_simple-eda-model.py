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
# Lets import data and print details about training dataset.



train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

display(train_df.info())

display(train_df.head())
# Given this is a classificatin problem looking at the data lets drop columns which wont add value to classification result.

# Drop Name, Ticket and PassangerId



cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Pclass', 'Cabin']

train_df = train_df.drop(cols_to_drop, axis=1)

test_df = test_df.drop(cols_to_drop, axis=1)



display(train_df.head())
display(train_df.isnull().sum())

display(test_df.isnull().sum())
train_df['Age'] = train_df['Age'].fillna(99)

test_df['Age'] = test_df['Age'].fillna(99)



# train_df['Cabin'] = train_df['Cabin'].fillna('Default')

# test_df['Cabin'] = test_df['Cabin'].fillna('Default')



train_df['Embarked'] = train_df['Embarked'].fillna('Default')

test_df['Embarked'] = test_df['Embarked'].fillna('Default')



test_df['Fare'] = test_df['Fare'].fillna(0)



display(train_df.head())
display(train_df.isnull().sum())

display(test_df.isnull().sum())
# Label encode non-numerical cols



from sklearn.preprocessing import LabelEncoder



# le = LabelEncoder()

# display(train_df.apply(le.fit_transform))

# test_df.apply(le.transform)



cols_to_encode = ['Sex', 'Embarked']

for col in cols_to_encode:

    

    print('col to encode - ', col)

    le = LabelEncoder()

    le = le.fit(train_df[col])

    train_df[col] = le.transform(train_df[col])

    test_df[col] = le.transform(test_df[col])



# col = 'Sex'

# display(train_df[col].head())

display(train_df.head())

display(test_df.head())
from matplotlib import pyplot as plt

import seaborn as sns

plt.figure(figsize=(16,6))



target_0 = train_df[train_df['Survived'] == 0]

target_1 = train_df[train_df['Survived'] == 1]



ax = sns.distplot(target_0[['Age']], hist=False, rug=True)

ax = sns.distplot(target_1[['Age']], hist=False, rug=True)

# sns.distplot(target_2[['SibSp']], hist=False, rug=True)
plt.figure(figsize=(16,6))

ax = sns.distplot(target_0[['Fare']], hist=False, rug=True)

ax = sns.distplot(target_1[['Fare']], hist=False, rug=True)
import seaborn as sns

sns.pairplot(train_df)
target_df = train_df['Survived']

input_df = train_df.drop(['Survived'], axis = 1)

X_submission = test_df
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(input_df, target_df, test_size=0.33, random_state=42)
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn import metrics



def get_model_NB(X, target, alpha=1):

    model = MultinomialNB(alpha=alpha).fit(X, target)

    return model



def get_model_RF(X, target):

    model = RandomForestClassifier()

    return model.fit(X, target)



def get_model_XGB_simple(X, target):

    model = xgb.XGBClassifier()

    return model.fit(X, target)

print(' ----- RANDOM FOREST ----- ')

model = get_model_RF(X_train, Y_train)

predicted = model.predict(X_test)

print(metrics.classification_report(Y_test, predicted))
print(' ----- MultinomialNB ----- ')

model = get_model_NB(X_train, Y_train)

predicted = model.predict(X_test)

print(metrics.classification_report(Y_test, predicted))
print(' ----- XGBOOST ----- ')

model = get_model_XGB_simple(X_train, Y_train)

predicted = model.predict(X_test)

print(metrics.classification_report(Y_test, predicted))