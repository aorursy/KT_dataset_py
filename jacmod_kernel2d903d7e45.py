import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#visualization 

from matplotlib import pyplot as plt

import seaborn as sns



#machine learning models

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

tr_d = train_data.copy()

te_d = test_data.copy()

train_data.head()

tr_d.describe(include = 'all')
tr_d.info()

te_d.info()
tr_d.mean()
tr_d['Embarked'].value_counts()
sns.barplot(x="Embarked", y="Survived", data=tr_d)
tr_d[['Survived', 'Embarked']].groupby('Embarked').mean().sort_values(by = 'Survived', ascending = False)
tr_d['Fare'].value_counts()
plt.hist([tr_d[tr_d['Survived'] == 1]['Fare'], 

          tr_d[tr_d['Survived'] == 0]['Fare']], 

         stacked=True, color = ['b','r'],

         bins = 25, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend()
tr_d[['Survived', 'Fare']].groupby('Fare').mean().sort_values(by = 'Survived', ascending = False)
tr_d['Parch'].value_counts()
sns.barplot(x="Parch", y="Survived", data=tr_d)
tr_d[['Survived', 'Parch']].groupby('Parch').mean().sort_values(by = 'Survived', ascending = False)
tr_d['SibSp'].value_counts()
sns.barplot(x="SibSp", y="Survived", data=tr_d)
tr_d[['SibSp', 'Survived']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False)
tr_d['Age'].value_counts()
plt.hist([tr_d[tr_d['Survived'] == 1]['Age'], 

          tr_d[tr_d['Survived'] == 0]['Age']], 

         stacked=True, color = ['b','r'],

         bins = 25, label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Number of passengers')

plt.legend()
tr_d[['Age', 'Survived']].groupby('Age').mean().sort_values(by='Survived', ascending=False)
tr_d['Sex'].value_counts()
sns.barplot(x="Sex", y="Survived", data=tr_d)
tr_d[['Sex', 'Survived']].groupby('Sex').mean().sort_values(by='Survived', ascending=False)
tr_d['Pclass'].value_counts()
sns.barplot(x='Pclass', y='Survived', data = tr_d)
tr_d[['Pclass', 'Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False)
#tr_d['Sex'] = tr_d['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

#te_d['Sex'] = te_d['Sex'].map( {'male': 0, 'female': 1} ).astype(int)



#tr_d['Embarked'] = tr_d['Embarked'].fillna('C')

#tr_d['Embarked'] = tr_d['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})



#te_d['Embarked'] = te_d['Embarked'].fillna('C')

#te_d['Embarked'] = te_d['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})



#tr_d['Age'] = tr_d['Age'].fillna(tr_d['Age'].median())

#te_d['Age'] = te_d['Age'].fillna(tr_d['Age'].median())





all_data = [tr_d, te_d]

drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']



for data in all_data:

    data.drop(drop_columns, inplace = True, axis = 1)

    data['Sex'] = data['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

    

    data['Embarked'] = data['Embarked'].fillna('C')

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    data['Age'] = data['Age'].fillna(tr_d['Age'].median())

 
te_d['Fare'] = te_d['Fare'].fillna(te_d['Fare'].median())
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(tr_d[['Pclass', 'Age', 'Parch', 'Fare']])

tr_d[['Pclass', 'Age', 'Parch', 'Fare']]  = scaler.transform(tr_d[['Pclass', 'Age', 'Parch', 'Fare']])

te_d[['Pclass', 'Age', 'Parch', 'Fare']] = scaler.transform(te_d[['Pclass', 'Age', 'Parch', 'Fare']])

tr_d.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    tr_d.drop('Survived', axis = 1), tr_d['Survived'], test_size=0.15, stratify = tr_d['Survived'])

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score
param_grid = { 

    "criterion" : ["gini", "entropy"], 

    "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 

    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35]}



dt = DecisionTreeClassifier()

clf_dt = GridSearchCV(dt, param_grid, n_jobs=-1)

clf_dt.fit(X_train, y_train)

print(clf_dt.best_score_)

print(clf_dt.best_estimator_)
final_dt = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf = 1, min_samples_split = 18)

final_dt.fit(X_train, y_train)

dt_pred = final_dt.predict(X_test)

accuracy_score(y_test, dt_pred)

param_grid = { 

    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None], 

    "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 

    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],

    'n_estimators': [5, 10, 20, 30, 40]}



rfc = RandomForestClassifier()

clf_rfc = GridSearchCV(rfc, param_grid, n_jobs=-1)

clf_rfc.fit(X_train, y_train)

print(clf_rfc.best_score_)

print(clf_rfc.best_estimator_)
final_rfc = RandomForestClassifier(max_depth = 10, min_samples_leaf = 5, min_samples_split = 18, n_estimators = 10)

final_rfc.fit(X_train, y_train)

rfc_pred = final_rfc.predict(X_test)

accuracy_score(y_test, rfc_pred)
param_grid = { 

    'penalty': ['l1', 'l2'], 

    "C" : [1, 0.1, 0.001, 0.0001], 

    "solver" : ['lbfgs', 'liblinear']}



lr = LogisticRegression()

clf_lr = GridSearchCV(lr, param_grid, n_jobs=-1)

clf_lr.fit(X_train, y_train)

print(clf_lr.best_score_)

print(clf_lr.best_estimator_)
final_lr = LogisticRegression(C = 0.1, penalty = 'l2', solver = 'liblinear')

final_lr.fit(X_train, y_train)

lr_pred = final_lr.predict(X_test)

accuracy_score(y_test, lr_pred)
param_grid = { 

    'C' : [ 0.1, 1, 10],

    'gamma' :  [0.1, 1],

    'kernel' : ['rbf', 'sigmoid']}



sv = SVC()

clf_sv = GridSearchCV(sv, param_grid, n_jobs=-1)

clf_sv.fit(X_train, y_train)

print(clf_sv.best_score_)

print(clf_sv.best_estimator_)
final_svm = SVC(C = 1, gamma = 0.1, kernel = 'rbf')

final_svm.fit(X_train, y_train)

svm_pred = final_svm.predict(X_test)

accuracy_score(y_test, svm_pred)
predictions = final_rfc.predict(te_d)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)