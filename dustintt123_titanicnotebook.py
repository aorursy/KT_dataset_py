# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train = train_df.copy()

test = test_df.copy()



# Change From Here



# Drop features

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test = test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



# Mapping features to numerical features

# Mapping Sex

mapping = {'male': 0, 'female': 1}

train = train.replace({'Sex': mapping})

test = test.replace({'Sex': mapping})

# Mapping Embarked

mapping = {'C': 3, 'Q': 2, 'S': 1}

train = train.replace({'Embarked': mapping})

test = test.replace({'Embarked': mapping})



# Engineered mapping by clusters

# Mapping Pclass to two categories

# For Female

mapping = {2: 1, 3: 0}

#train[train.Sex == 1] = train[train.Sex == 1].replace({'Pclass': mapping})

#test[test.Sex == 1] = test[test.Sex == 1].replace({'Pclass': mapping})

# For Male

mapping = {2: 0, 3: 0}

#train[train.Sex == 0] = train[train.Sex == 0].replace({'Pclass': mapping})

#test[test.Sex == 0] = test[test.Sex == 0].replace({'Pclass': mapping})



# Fill null for Age

train['Age'] = train['Age'].fillna(train['Age'].mean())

test['Age'] = test['Age'].fillna(test['Age'].mean())



# Fill the Embarked null values with mode

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

test['Embarked'] = test['Embarked'].fillna(test['Embarked'].mode()[0])



# Normalize

train['Fare'] = (train['Fare'] - train['Fare'].mean()) / train['Fare'].std()

train['Age'] = (train['Age'] - train['Age'].mean()) / train['Age'].std()

test['Fare'] = (test['Fare'] - test['Fare'].mean()) / test['Fare'].std()

test['Age'] = (test['Age'] - test['Age'].mean()) / test['Age'].std()



# Seperate the X, Y in training set

train_X = train.iloc[:, 1:]

train_Y = train.iloc[:, 0]





# Fill null in test

test = test.fillna(test.mean())



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_X, train_Y)

Y_pred = random_forest.predict(test)

random_forest.score(train_X, train_Y)

acc_random_forest = round(random_forest.score(train_X, train_Y) * 100, 2)

acc_random_forest





# Change End Here



# train['FamilySize'] = train['SibSp'] + train['Parch']

# train[['FamilySize', 'Sex', 'Survived']].groupby([ 'Sex','FamilySize'], as_index=False).mean()

# train[['FamilySize', 'Sex', 'Survived']].groupby([ 'Sex','FamilySize'], as_index=False).count()

# train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()





#check = train[train.Pclass == 3]

#check = check[train.Sex == 1]

#check.describe()





# train[['Sex', 'Pclass', 'Survived']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


