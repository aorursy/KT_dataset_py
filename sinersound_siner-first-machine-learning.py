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
# data analysis and wrangling
import keras
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
combine = [train_df, test_df]
test_df
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_df['AgeBand'] = pd.cut(train_df['Age'], 10)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
train_df['Age'].fillna(train_df['Age'].dropna().median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 8, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 8) & (dataset['Age'] <= 16), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 24), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 32), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 40), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 48), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 56), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 56) & (dataset['Age'] <= 64), 'Age'] = 7
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 72), 'Age'] = 8
    dataset.loc[(dataset['Age'] > 72) & (dataset['Age'] <= 80), 'Age'] = 9
    dataset.loc[dataset['Age'] > 80, 'Age'] = 10
    
    dataset['Age'] = dataset['Age'].astype(int)
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]
train_df.head()
train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
combine = [train_df, test_df]

train_df.head()
train_df = train_df.drop(['Sex'], axis=1)
test_df = test_df.drop(['Sex'], axis=1)
combine = [train_df, test_df]
train_df['Embarked'].fillna('S', inplace=True)
test_df['Embarked'].fillna('S', inplace=True)

test_df

for dataset in combine:
    dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 0
    dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 1
    dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 2
    
combine = [train_df, test_df]
train_df.head()
test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 20)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[dataset['Fare'] <= 10, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 20), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 30), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 40), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 50), 'Fare'] = 4
    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 70), 'Fare'] = 5
    dataset.loc[(dataset['Fare'] > 70) & (dataset['Fare'] <= 90), 'Fare'] = 6
    dataset.loc[(dataset['Fare'] > 90) & (dataset['Fare'] <= 100), 'Fare'] = 7
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 8
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df.head()
train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
for dataset in combine:
    dataset.loc[dataset['PassengerId'] <= 50, 'PassengerIndex'] = 0
    dataset.loc[(dataset['PassengerId'] > 50) & (dataset['PassengerId'] <= 100), 'PassengerIndex'] = 1
    dataset.loc[(dataset['PassengerId'] > 100) & (dataset['PassengerId'] <= 150), 'PassengerIndex'] = 2
    dataset.loc[(dataset['PassengerId'] > 150) & (dataset['PassengerId'] <= 200), 'PassengerIndex'] = 3
    dataset.loc[(dataset['PassengerId'] > 200) & (dataset['PassengerId'] <= 300), 'PassengerIndex'] = 4
    dataset.loc[(dataset['PassengerId'] > 300) & (dataset['PassengerId'] <= 400), 'PassengerIndex'] = 5
    dataset.loc[(dataset['PassengerId'] > 400) & (dataset['PassengerId'] <= 500), 'PassengerIndex'] = 6
    dataset.loc[(dataset['PassengerId'] > 500) & (dataset['PassengerId'] <= 600), 'PassengerIndex'] = 7
    dataset.loc[(dataset['PassengerId'] > 600) & (dataset['PassengerId'] <= 700), 'PassengerIndex'] = 8
    dataset.loc[(dataset['PassengerId'] > 700) & (dataset['PassengerId'] <= 800), 'PassengerIndex'] = 9
    dataset.loc[(dataset['PassengerId'] > 800) & (dataset['PassengerId'] <= 900), 'PassengerIndex'] = 10
    dataset.loc[dataset['PassengerId'] > 900, 'PassengerIndex'] = 11

combine = [train_df, test_df]
test_df
X_train = train_df.drop(['Survived', 'PassengerId', 'PassengerIndex'], axis=1)
Y_train = train_df['Survived']
X_test = test_df.copy()
X_test = X_test.drop(['PassengerId', 'PassengerIndex'], axis=1)
Y_test = test_df['PassengerId']
X_train.shape, Y_train.shape, X_test.shape
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
Y_pred
X_test
print(Y_test)
result = list()
for i, preds in enumerate(Y_pred):
    result.append([Y_test[i], preds])
    
sub = pd.DataFrame(result, columns=['PassengerId', 'Survived'])
sub.to_csv('sample_submission.csv', index=False)
