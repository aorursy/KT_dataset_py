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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
train.hist(figsize=(12,12))

plt.show()
sns.heatmap(train.corr(), square = True, vmax = .8)

plt.show()
train.describe()
train.drop(['PassengerId'], inplace = True, axis = 1)
test.drop(['PassengerId'], inplace = True, axis = 1)
print("Cabins:")

print(train['Cabin'].unique())

print("\nGenders:")

print(train['Sex'].unique())

print("\nEmbarked:")

print(train['Embarked'].unique())
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

data = [train, test]



for dataset in data:

    dataset['Cabin'] = dataset['Cabin'].fillna("U0")

    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())

    dataset['Deck'] = dataset['Deck'].map(deck)

    dataset['Deck'] = dataset['Deck'].fillna(0)

    dataset['Deck'] = dataset['Deck'].astype(int)



train.drop(['Cabin'], axis=1, inplace = True)

test.drop(['Cabin'], axis=1, inplace = True)
data = [train, test]



for dataset in data:

    mean = train["Age"].mean()

    std = test["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train["Age"].astype(int)

    

train["Age"].isnull().sum()
train['Embarked'].describe()
common_value = 'S'

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)
train.info()
data = [train, test]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
genders = {"male": 0, "female": 1}

data = [train, test]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
train.drop(['Ticket'], axis = 1, inplace = True)

test.drop(['Ticket'], axis = 1, inplace = True)
ports = {"S": 0, "C": 1, "Q": 2}

data = [train, test]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
data = [train, test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    # convert titles into numbers

    dataset['Title'] = dataset['Title'].map(titles)

    # filling NaN with 0, to get safe

    dataset['Title'] = dataset['Title'].fillna(0)



train.drop(['Name'], axis=1, inplace = True)

test.drop(['Name'], axis=1, inplace = True)
"""data = [train, test]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6"""
"""data = [train, test]



for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)"""
data = [train, test]

for dataset in data:

    dataset['Age_Class']= dataset['Age']* dataset['Pclass']
train.head(10)
sns.heatmap(train.corr())
X_train = train.drop(['Survived'], axis = 1)

Y_train = train['Survived']

X_test = test.copy()
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

accuracy = random_forest.score(X_train, Y_train) * 100
print("Accuracy: {}".format(accuracy))
from sklearn.metrics import classification_report
classification_report(Y_train, random_forest.predict(X_train))
Y_pred
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(submission.head(10))
submission['Survived'] = Y_pred
print(submission.head(10))
submission.to_csv('titanic_submission.csv',index=False)
print(len(submission))
random_forest = RandomForestClassifier(min_samples_leaf = 3, 

                                       min_samples_split = 10,   

                                       n_estimators=200, 

                                       max_features=0.5,

                                       n_jobs=-1)



random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

accuracy = random_forest.score(X_train, Y_train) * 100



print("Accuracy: {}".format(accuracy))

#print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
submission['Survived'] = Y_pred
submission.to_csv('titanic_submission.csv',index=False)
print(len(Y_pred))