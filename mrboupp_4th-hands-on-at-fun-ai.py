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
from sklearn import preprocessing
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



test_df = pd.read_csv("../input/titanic/test.csv")
train['Cabin'].fillna(train['Cabin'].mode()[0],inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

train['Age'].fillna(train['Age'].mode()[0],inplace=True)

train.drop(columns=['Name','Ticket'],inplace=True)

train['Fare'].fillna(train['Fare'].mode()[0],inplace=True)



for column in ['Sex','Cabin','Embarked']:

    le = preprocessing.LabelEncoder()

    le.fit(train[column])

    train[column] = le.transform(train[column])

    

full_data = [train]

for train in full_data:

    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1



train.drop(columns=['SibSp','Parch'],inplace=True)

train.drop(columns='PassengerId',inplace=True)

train.info()
test['Cabin'].fillna(test['Cabin'].mode()[0],inplace=True)

test['Embarked'].fillna(test['Embarked'].mode()[0],inplace=True)

test['Age'].fillna(test['Age'].mode()[0],inplace=True)

test.drop(columns=['Name','Ticket'],inplace=True)

test['Fare'].fillna(test['Fare'].mode()[0],inplace=True)



for column in ['Sex','Cabin','Embarked']:

    le = preprocessing.LabelEncoder()

    le.fit(test[column])

    test[column] = le.transform(test[column])



full_data = [test]

for test in full_data:

    test['FamilySize'] = test['SibSp'] + test['Parch'] + 1



test.drop(columns=['SibSp','Parch'],inplace=True)

test.drop(columns='PassengerId',inplace=True)

test.info()

train.head()
from sklearn.model_selection import train_test_split

Y_train = train['Survived']

X_train = train.drop(['Survived'], axis=1)



X_train , X_test , Y_train , Y_test = train_test_split(X_train , Y_train ,test_size = 0.1,random_state = 0)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

acc_svc = round(svc.score(X_test, Y_test) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

acc_knn = round(knn.score(X_test, Y_test) * 100, 2)

acc_knn
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

acc_gaussian = round(gaussian.score(X_test, Y_test) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

acc_perceptron = round(perceptron.score(X_test, Y_test) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

acc_sgd = round(sgd.score(X_test, Y_test) * 100, 2)

acc_sgd


decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)

acc_random_forest
Y_pred = random_forest.predict(test)
Y_pred
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('submission.csv', index=False)