# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import data and review 

train = pd.read_csv(r'/kaggle/input/titanic/train.csv')

train.head()

test = pd.read_csv(r'/kaggle/input/titanic/test.csv')

test.head()

train['Name'].unique()
# Get Title from Name

train['Title'] = train['Name'].str.split(', ').apply(lambda x: x[1].split('.')[0])

test['Title'] = test['Name'].str.split(', ').apply(lambda x: x[1].split('.')[0])

train.head()
# Get Family Size

train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + test['Parch']

train.head(10)
# Get average age by title

train['Title'].value_counts()

titleMap = {"Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master", "Dr": "Dr", "Rev": "Rev", "Mlle": "Mrs", "Col": "Mr", "Major": "Mr", "Jonkheer": "Mr", "Capt": "Mr", "Don": "Mr", "Sir": "Mr", 

           "Mme": "Miss", "Ms": "Miss", "Lady": "Miss", "the Countess": "Mrs"}

train['Title'] = train['Title'].map(titleMap)

train['Title'] = train['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5})

test['Title'] = test['Title'].map(titleMap)

test['Title'] = test['Title'].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5})
test['Title'].fillna(2, inplace=True)

test['Fare'].fillna(test['Fare'].mean(), inplace=True)

test.describe()
#Get average age by title and fill na

train['Age'].isna().sum()

meanAge = train['Age'].groupby(by=train['Title']).mean()

meanAge
train['Age'].fillna(train.groupby(['Title'])['Age'].transform(np.mean), inplace=True)

test['Age'].fillna(test.groupby(['Title'])['Age'].transform(np.mean), inplace=True)

train.head()
#Code Sex

train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)

train.head()
#Embarked

train[train.Embarked.isna()]

train['Embarked'].value_counts() #S is most common

train['Embarked'].fillna('S', inplace=True)

train['Embarked'] = train['Embarked'].map({"S": 0, "C": 1, "Q": 2})

test['Embarked'].fillna('S', inplace=True)

test['Embarked'] = test['Embarked'].map({"S": 0, "C": 1, "Q": 2})

train.head()

#Feature for Alone

train['Alone'] = train['FamilySize'].apply(lambda x: 1 if x == 0 else 0)

test['Alone'] = test['FamilySize'].apply(lambda x: 1 if x == 0 else 0)
#Remove unneeded features

train = train.drop(columns=['Cabin', 'Name', 'Ticket'])

#Model

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Title', 'FamilySize', 'Alone']

target = ['Survived']





X_train = train[features]

Y_train = train['Survived']

X_test = test[features]



X_train.shape, Y_train.shape, X_test.shape

X_test.describe()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

log_accuracy = round(logreg.score(X_train, Y_train) * 100, 2)

log_accuracy
estimators = range(1,100, 5)

acc = []

for estimator in estimators:

    random_forest = RandomForestClassifier(n_estimators=estimator)

    random_forest.fit(X_train, Y_train)

    Y_pred = random_forest.predict(X_test)

    random_forest.score(X_train, Y_train)

    rf_accuracy = round(random_forest.score(X_train, Y_train) * 100, 2)

    acc.append(rf_accuracy)

acc

clfSVC = LinearSVC().fit(X_train, Y_train)

Y_pred = clfSVC.predict(X_test)

accSVC = round(clfSVC.score(X_train, Y_train)*100, 2)



clflog = LogisticRegression(C=50).fit(X_train, Y_train)

Y_pred = clflog.predict(X_test)

acclog = round(clflog.score(X_train, Y_train)*100, 2)



print(accSVC, acclog)
tree = DecisionTreeClassifier(max_depth = 4, random_state=0).fit(X_train, Y_train)

Y_predtree = tree.predict(X_test)

acctree = round(tree.score(X_train, Y_train)*100, 2)

acctree
n_neighbors = range(1,10)

for neighbor in n_neighbors:

    clf = KNeighborsClassifier(n_neighbors=neighbor)

    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)

    acc = round(clf.score(X_train, Y_train)*100, 2)

    print(neighbor, acc)

#Save output

predict = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_predtree})

predict.to_csv('kastner_submission.csv', index=False)