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
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

test2_df = pd.read_csv("/kaggle/input/titanic/test.csv")
# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

full_data = [train_df, test_df]

for dataset in  full_data:

    dataset['Family Size'] = dataset['SibSp'] + dataset['Parch'] + 1

    

    
for dataset in full_data:

    dataset['isAlone'] = 0

    dataset.loc[dataset['Family Size'] == 1, 'isAlone'] = 1

    #JO bhi akela h usko 1 Dediya and Baaki Jo bhi Family k Saath h unko 0
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

    #fills missing values in Embarked
for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())

train_df['Categorical Fare'] = pd.qcut(train_df['Fare'], 4)

# Yh Fair ko 4 Categories m bantt tha h
for dataset in full_data:

    # extract titles

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    # replace titles with a more common title or as Rare

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')




for dataset in full_data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

train_df["Age"].isnull().sum()
for dataset in full_data:

    sex_map = {'female':0, 'male': 1}

    dataset['Sex'] = dataset['Sex'].fillna(0)

    dataset['Sex'] = dataset['Sex'].map(sex_map).astype(int)

    

    title_map = {'Mr': 1, 'Miss': 2, 'Mrs': 3, "Master": 4, 'Rare': 5}

    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Title'] = dataset['Title'].map(title_map).astype(int)

    

    

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

    

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    

    

    

    

    

    
for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
drop_elements = ['Name', 'Family Size', 'Ticket', 'Cabin', 'SibSp', 'Parch','PassengerId']

train_df = train_df.drop(drop_elements, axis = 1)

train_df = train_df.drop('Categorical Fare', axis = 1)

test_df = test_df.drop(drop_elements, axis = 1)
print(test_df.head(10))
print(train_df.head(10))
X_train = train_df.drop('Survived', axis = 1).copy()

Y_train = train_df['Survived']

X_test = test_df.copy()
train_df  = train_df.drop("isAlone", axis=1)

test_df  = test_df.drop("isAlone", axis=1)



train_df  = train_df.drop("Parch", axis=1)

test_df  = test_df.drop("Parch", axis=1)
random_forest = RandomForestClassifier(n_estimators=100, random_state = 1)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_test, Y_prediction) * 100, 2)







output = pd.DataFrame({'PassengerId': test2_df.PassengerId, 'Survived': Y_prediction})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



print(round(random_forest.score(X_train, Y_train) * 100, 2))

random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)

random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)



acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())