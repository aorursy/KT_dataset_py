# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn import tree

import matplotlib.pyplot as plt

import numpy as np

import csv

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')
#Encode gender

le = preprocessing.LabelEncoder()

le.fit(train_data.Sex)

train_data.Sex = le.transform(train_data.Sex)

test_data.Sex = le.transform(test_data.Sex)
a = test_data.Fare.fillna(test_data.Fare.mean())

test_data.Fare =a
#Predict Age

linreg = LinearRegression()

data = train_data[['Survived','Pclass','Sex','SibSp','Parch','Fare','Age']]

data2 = test_data[['Pclass','Sex','SibSp','Parch','Fare','Age']]
x_train = data[data['Age'].notnull()].drop(columns='Age')

y_train = data[data['Age'].notnull()]['Age']

x_test = data[data['Age'].isnull()].drop(columns='Age')

y_test = data[data['Age'].isnull()]['Age']



x_t_train = data2[data2['Age'].notnull()].drop(columns='Age')

y_t_train = data2[data2['Age'].notnull()]['Age']

x_t_test = data2[data2['Age'].isnull()].drop(columns='Age')

y_t_test = data2[data2['Age'].isnull()]['Age']
linreg = LinearRegression()

linreg.fit(x_train,y_train)

predicted = linreg.predict(x_test)

train_data.Age[train_data.Age.isnull()] = predicted





lin= LinearRegression()

lin.fit(x_t_train,y_t_train)

predicted = lin.predict(x_t_test)

test_data.Age[test_data.Age.isnull()] = predicted
Y1 = train_data.Survived

X1 = train_data.drop(columns=['Survived','PassengerId','Name','Ticket','Embarked','Cabin'],axis = 1)
k_range = list(range(1, 31))

weight_options = ['uniform','distance']

param_grid = dict(n_neighbors=k_range, weights=weight_options)
grid = GridSearchCV(KNeighborsClassifier(),param_grid,cv=10,scoring = "accuracy",return_train_score=False)

grid.fit(X1,Y1)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']][:3]
print(grid.best_score_)

print(grid.best_params_)
penalty_options = ['l2']

param_grid = dict(penalty =penalty_options)
grid = GridSearchCV(LogisticRegression(),param_grid, cv=10,scoring = "accuracy",return_train_score=False)

grid.fit(X1,Y1)
pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
clf = tree.DecisionTreeClassifier()

param_grid={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
grid = GridSearchCV(clf,param_grid,cv=10,scoring = "accuracy",return_train_score=False)

grid.fit(X1,Y1)
print(grid.best_score_)

print(grid.best_params_)
clf = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split= 30)

clf = clf.fit(X1,Y1)

pred_Y1 = clf.predict(X1)

accuracy_score = accuracy_score(Y1, pred_Y1)

accuracy_score
X_test = test_data.drop(columns=['PassengerId','Name','Ticket','Embarked','Cabin'],axis = 1)

results = clf.predict(X_test)
results
lines = [['PassengerId', 'Survived']]

for i,j in enumerate(results):

    lines.append([i,j])

    

with open('accuracy0.8552.csv','w') as writeFile:

    writer = csv.writer(writeFile)

    writer.writerows(lines)



writeFile.close()