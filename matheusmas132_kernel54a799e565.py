# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

train.head()
train.info()
train = train.drop(columns=["Age", "Cabin", "Embarked", "Name", "Ticket"])
train.info()
train["Sex"] = train["Sex"].replace(to_replace = "female", value = "0")

train["Sex"] = train["Sex"].replace(to_replace = "male", value = "1")

train["Sex"] = train.Sex.astype('int64', copy=False)



test["Sex"] = test["Sex"].replace(to_replace = "female", value = "0")

test["Sex"] = test["Sex"].replace(to_replace = "male", value = "1")

test["Sex"] = test.Sex.astype('int64', copy=False)
train["Family"] = train["SibSp"] + train["Parch"]

test["Family"] = test["SibSp"] + test["Parch"]
train.info()
train.head()
train.describe()
a = sns.heatmap(train.corr(), annot=True)
test.Fare = test.Fare.fillna(35.627)
# from xgboost import XGBClassifier

# from sklearn.ensemble import GradientBoostingClassifier



# # Targets:

# Target = ["Fare", "Pclass", "Sex"]

# # Target = ["Fare", "Sex"]

# # Target = ["Fare", "Pclass", "Sex", "SibSp", "Parch"]



# train_X = train[Target]

# train_y = train["Survived"]

# test_X = test[Target]



# # model = GradientBoostingClassifier(max_depth=3, n_estimators=500, learning_rate=0.05)

# model = XGBClassifier(max_depth=2, n_estimators=500, learning_rate=0.5, verbosity=0, gamma=0.5, objective= 'binary:logistic', seed=27)

# model.fit(train_X, train_y)



# test_pred = model.predict(test_X)

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



Target = ["Fare", "Pclass", "Sex", "SibSp", "Parch", "Family"]



train_X = train[Target]

train_y = train["Survived"]

test_X = test[Target]



train_dmatrix = xgb.DMatrix(data = train_X, label = train_y)



gbm_param_grid = {'learning_rate': [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.5, 0.9],

                  'n_estimators': [200, 300, 400, 500, 600],

                  'max_depth': [2, 3, 4, 5, 8]}

gbm = xgb.XGBClassifier()

grid = GridSearchCV(estimator=gbm, param_grid = gbm_param_grid, scoring="accuracy", cv=4, verbose = 1, n_jobs=-1)



# rfc_param_grid = { 

#     'n_estimators': [200, 300, 400, 500, 600],

#     'max_features': ['auto', 'sqrt', 'log2'],

#     'max_depth' : [4,5,6,7,8],

#     'criterion' :['gini', 'entropy']

# }

# rfc = RandomForestClassifier(random_state=42)

# grid = GridSearchCV(estimator=rfc, param_grid=rfc_param_grid, scoring="accuracy", cv= 5, verbose = 1, n_jobs=-1)



grid.fit(train_X, train_y)

test_pred = grid.predict(test_X)



print(grid.best_estimator_)

print(grid.best_params_)

print(grid.best_score_)
submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived' : test_pred })

submission.to_csv("submission.csv", index=False)