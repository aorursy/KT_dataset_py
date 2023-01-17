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

import warnings

warnings.simplefilter('ignore')

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import re

import matplotlib.pyplot as plt
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

train_data.head()
probClass = train_data[['Pclass', 'Survived']].groupby('Pclass').agg(lambda x : sum(x == 1) / len(x))
train_data['Embarked'][train_data['Embarked'].isna()] = train_data['Embarked'].mode()[0]

probEmbarked = train_data[['Embarked', 'Survived']].groupby('Embarked').agg(lambda x : sum(x == 1) / len(x))
probSex = train_data[['Sex', 'Survived']].groupby('Sex').agg(lambda x : sum(x == 1) / len(x))
probSex
totalAge = train_data[train_data['Sex'] == 'male'].value_counts('Age').sort_index().to_frame()

survivedAge = train_data[(train_data['Sex'] == 'male') & (train_data['Survived'] == 1)].value_counts('Age').sort_index().to_frame()

totalAge['Age'] = np.round(totalAge.index / 10) * 10.

totalAge.index.rename(None, inplace=True)

totalAge.groupby('Age').sum()

survivedAge['Age'] = np.round(survivedAge.index / 10) * 10.

survivedAge.index.rename(None, inplace=True)

survivedAge.groupby('Age').sum()
ageProbTab = pd.merge(

    left = totalAge.groupby('Age').sum(), 

    right = survivedAge.groupby('Age').sum(), 

    right_index=True, 

    left_index=True,

    how = 'left'

).fillna(0)



ageProbTab['Prob'] = ageProbTab['0_y'] / ageProbTab['0_x']



def probAge(age):

    if np.isnan(age):

        return 0.23

    return ageProbTab['Prob'][round(age / 10) * 10]
train_data['Age_prob'] = train_data['Age']

train_data['Age_prob'][train_data['Sex'] == 'male'] = train_data['Age'][train_data['Sex'] == 'male'].apply(probAge)

train_data['Age_prob'][train_data['Sex'] == 'female'] = 0.99

train_data['Sex_prob'] = train_data['Sex'].apply(lambda x : probSex["Survived"][x])

train_data['Embarked_prob'] = train_data['Embarked'].apply(lambda x : probEmbarked["Survived"][x])

train_data['Pclass_prob'] = train_data['Pclass'].apply(lambda x : probClass["Survived"][x])
train_data
X = train_data[["Age_prob", "Sex_prob", "Embarked_prob", "Pclass_prob"]]

y = train_data["Survived"]

sum(~(y ^ (train_data["Sex"] == 'female'))) / len(y)
cross_val_score(DecisionTreeClassifier(), X, y, cv=5).mean()
cross_val_score(LogisticRegression(), X, y, cv=5).mean()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.info()
test_data['Age_prob'] = test_data['Age']

test_data['Age_prob'][test_data['Sex'] == 'male'] = test_data['Age'][test_data['Sex'] == 'male'].apply(probAge)

test_data['Age_prob'][test_data['Sex'] == 'female'] = 0.99

test_data['Sex_prob'] = test_data['Sex'].apply(lambda x : probSex["Survived"][x])

test_data['Embarked_prob'] = test_data['Embarked'].apply(lambda x : probEmbarked["Survived"][x])

test_data['Pclass_prob'] = test_data['Pclass'].apply(lambda x : probClass["Survived"][x])

test_data
y_test = DecisionTreeClassifier().fit(X, y).predict(test_data[["Age_prob", "Sex_prob", "Embarked_prob", "Pclass_prob"]])
output = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : y_test.astype(np.int32)})

output
output.to_csv('submission.csv', index=False)