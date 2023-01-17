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

import re

import matplotlib.pyplot as plt

df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

Titles = ['Mr', 'Ms', 'Mrs', 'Miss', 'Master', 

          'Don', 'Rev', 'Dr', 'Mme', 

          'Major', 'Lady', 'Sir', 'Mlle', 

          'Col', 'Capt', 'the Countess', 'Jonkheer']

reg = r'(?:(Mr|Mrs|Ms|Miss|Master|Don|Rev|Dr|Mme|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer)\.)'



def get_means(df):

    means = []

    for i in range(len(Titles)):

        means.append(df['Age'][(df['Age'] > 0) & (df['Title'] == i)].mean())

    return means



def get_title(x):

    ret = -1

    r = re.findall(reg, x, flags=re.IGNORECASE)

    if len(r):

        ret = Titles.index(r[0])

    return ret
df['Title'] = df['Name'].apply(get_title)

df['SexCode'] = df['Sex'].apply(lambda x: 0 if x == 'female' else 1)

df['Embarked'].fillna('S', inplace=True)

df['EmbCode'] = df['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))

df['Age'].fillna(-1, inplace=True)

train_age_means = get_means(df)

df['Age'] = df.apply(lambda row: row['Age'] if row['Age'] != -1 else train_age_means[row['Title']], axis=1) 

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Title', 'SexCode','EmbCode']],  df['Survived'], test_size=0.33, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

y_pred

# model.fit(df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Title', 'SexCode','EmbCode']],  df['Survived'])
model.fit(df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Title', 'SexCode','EmbCode']],  df['Survived'])

test_df['Title'] = test_df['Name'].apply(get_title)

test_df['SexCode'] = test_df['Sex'].apply(lambda x: 0 if x == 'female' else 1)

test_df['Embarked'].fillna('S', inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)

test_df['EmbCode'] = test_df['Embarked'].apply(lambda x: 0 if x == 'S' else (1 if x == 'C' else 2))

test_df['Age'].fillna(-1, inplace=True)

test_df['Age'] = test_df.apply(lambda row: row['Age'] if row['Age'] != -1 else train_age_means[row['Title']], axis=1)



y_test1 = model.predict(test_df[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Title', 'SexCode','EmbCode']])

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':y_test1})

submission.to_csv('submission.csv', index=False)