import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/titanic/train.csv")

train
test = pd.read_csv("/kaggle/input/titanic/test.csv")

test
y_train = train.pop('Survived')
train.info()
test.info()
train = train.drop(labels=['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

test = test.drop(labels=['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
train['Embarked'].unique()
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace = True)

test['Embarked'].fillna('S', inplace = True)
train['Fare'].fillna(train['Fare'].mean(), inplace = True)

train['Age'].fillna(train['Age'].mean(), inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)
dummy = pd.get_dummies(train['Sex'], drop_first= True)

train = pd.concat((train, dummy), axis = 1)

train = train.drop(labels = 'Sex', axis = 1)

dummy = pd.get_dummies(train['Embarked'], drop_first= True)

train = pd.concat((train,dummy), axis = 1)

train = train.drop(labels = 'Embarked', axis = 1)

train
dummy = pd.get_dummies(test['Sex'], drop_first= True)

test = pd.concat((test, dummy), axis = 1)

test = test.drop(labels = 'Sex', axis = 1)

dummy = pd.get_dummies(test['Embarked'], drop_first= True)

test = pd.concat((test,dummy), axis = 1)

test = test.drop(labels = 'Embarked', axis = 1)

test
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=1)

classifier.fit(train,y_train)
y_pred = classifier.predict(test)
y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col = 'PassengerId')
from sklearn.metrics import accuracy_score

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.svm import SVC

classifier = SVC(random_state=1)

classifier.fit(train,y_train)

y_pred = classifier.predict(test)

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(train,y_train)

y_pred = classifier.predict(test)

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=1)

classifier.fit(train,y_train)

y_pred = classifier.predict(test)

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(train,y_train)

y_pred = classifier.predict(test)

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.model_selection import GridSearchCV

parameters = [{'criterion': ['gini', 'entropy'], 'n_estimators' : [10,100,1000], 'max_depth' : [3,5,10]}]

grid_search = GridSearchCV(estimator = classifier,

                          param_grid = parameters,

                          scoring = 'accuracy',

                          cv = 10,

                          n_jobs = -1)

grid_search = grid_search.fit(train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate= 0.01, max_depth= 3, n_estimators=100)

classifier.fit(train, y_train)

y_pred = classifier.predict(test)

from sklearn.metrics import accuracy_score

ac = accuracy_score(y_pred, y_test)

ac
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred, y_test)

cm