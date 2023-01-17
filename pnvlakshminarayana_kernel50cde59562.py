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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.head(2))

print(test.head(2))

train.dtypes
train["Sex"] = train["Sex"].astype('category')

train["Embarked"] = train["Embarked"].astype('category')

train.dtypes
null_columns=train.columns[train.isnull().any()]

train[null_columns].isnull().sum()



train["Embarked"].value_counts()
Age_mean=train["Age"].mean()

train["Age"].fillna(Age_mean, inplace = True) 



train["Embarked"].fillna('S', inplace = True) 
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

train["Sex_ecode"] = lb_make.fit_transform(train["Sex"])

train[["Sex", "Sex_ecode"]].head(3)



train["Embarked_ecode"] = lb_make.fit_transform(train["Embarked"])

train[["Embarked", "Embarked_ecode"]].head(10)
trainfinal=train[["PassengerId","Pclass","Survived","Age","Sex_ecode","SibSp","Parch","Fare","Embarked_ecode"]]

trainfinal.head()
trainlabel = trainfinal['Survived']

trainfinal = trainfinal.drop(['Survived'],axis = 1)



print(trainfinal.head())

print("*"*75)

print(trainlabel.head())
from sklearn.linear_model import LogisticRegression

clfl = LogisticRegression(C=0.1);

clfl.fit(trainfinal, trainlabel)



from sklearn.metrics import classification_report

print("LR train accuracy:",round(clfl.score(trainfinal,trainlabel)*100,2))
from xgboost import XGBClassifier

from sklearn import model_selection

clfx = XGBClassifier()



clfx.fit(trainfinal, trainlabel)

print("XGB train accuracy:",round(clfx.score(trainfinal,trainlabel)*100,2))
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV



estimator = XGBClassifier()



parameters = { "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }



grid_search_xgb = GridSearchCV(

    estimator=estimator,

    param_grid=parameters,

    scoring = 'accuracy',

    cv = 2,

    n_jobs=10,

    verbose=True

)
grid_search_xgb.fit(trainfinal, trainlabel)
print("XGB train accuracy with param tuning:",round(grid_search_xgb.score(trainfinal,trainlabel)*100,2))



grid_search_xgb.best_params_
test["Sex"] = test["Sex"].astype('category')

test["Embarked"] = test["Embarked"].astype('category')



test["Sex_ecode"] = lb_make.fit_transform(test["Sex"])

test[["Sex", "Sex_ecode"]].head(3)



test["Embarked_ecode"] = lb_make.fit_transform(test["Embarked"])

test[["Embarked", "Embarked_ecode"]].head(10)



Age_mean=test["Age"].mean()

test["Age"].fillna(Age_mean, inplace = True) 



test["Embarked"].fillna('S', inplace = True) 
null_columns=test.columns[test.isnull().any()]

test[null_columns].isnull().sum()
Fare_mean=test["Fare"].mean()

test["Fare"].fillna(Fare_mean, inplace = True) 
null_columns=test.columns[test.isnull().any()]

test[null_columns].isnull().sum()
testfinal=test[["PassengerId","Pclass","Age","Sex_ecode","SibSp","Parch","Fare","Embarked_ecode"]]

testfinal.head()
pred = grid_search_xgb.predict(testfinal)

print(pred)
submission = pd.DataFrame({

        "PassengerId": testfinal["PassengerId"],

        "Survived": pred

    })



submission.to_csv('submission.csv', index=False)