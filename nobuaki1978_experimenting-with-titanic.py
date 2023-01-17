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
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

test_df=pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
#train_df.isnull().sum()

train_df['Age'].fillna(train_df['Age'].mean(),inplace=True)

train_df['Cabin'].fillna(train_df['Cabin'].mode()[0],inplace=True)

train_df['Fare'].fillna(train_df['Fare'].mean(),inplace=True)

test_df.isnull().sum()
test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)

test_df['Cabin'].fillna(test_df['Cabin'].mode()[0],inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)

test_df.isnull().sum()
drop_column = ['Cabin']

train_df.drop(drop_column, axis=1, inplace = True)

test_df.drop(drop_column,axis=1,inplace=True)
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, classification_report

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.linear_model import LinearRegression
train.dropna(inplace=True)

relevant = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

y_train = train.Survived.copy()

X_train = train[relevant].copy()

dum_sex = pd.get_dummies(train.Sex).copy()

dum_emb = pd.get_dummies(train.Embarked).copy()



#dum_sex.head()

#dum_emb.head()

X_train.drop(['Sex','Embarked'], inplace=True, axis=1)

X_train = pd.concat([X_train, dum_sex, dum_emb], axis=1).copy()

regr_tree = DecisionTreeRegressor(max_depth=4,random_state=24)

regr_tree.fit(X_train,y_train)

results = pd.DataFrame(index=X_train.columns, data={'importance':regr_tree.feature_importances_})

#print('Feature importances:\n{}'.format(results))

score = regr_tree.score(X_train,y_train)

# Verification:

print('The r2 score is {:5.3f}.'.format(score))

regr_tree = DecisionTreeRegressor(max_depth=10,random_state=24)

regr_tree.fit(X_train,y_train)

results = pd.DataFrame(index=X_train.columns, data={'importance':regr_tree.feature_importances_})

score = regr_tree.score(X_train,y_train)

print('The r2 score is {:5.3f}.'.format(score))
regr_tree = DecisionTreeRegressor(max_depth=7,random_state=24)

regr_tree.fit(X_train,y_train)

results = pd.DataFrame(index=X_train.columns, data={'importance':regr_tree.feature_importances_})

score = regr_tree.score(X_train,y_train)

print('The r2 score is {:5.3f}.'.format(score))
regr_tree = DecisionTreeRegressor(max_depth=12,random_state=24)

regr_tree.fit(X_train,y_train)

results = pd.DataFrame(index=X_train.columns, data={'importance':regr_tree.feature_importances_})

score = regr_tree.score(X_train,y_train)

print('The r2 score is {:5.3f}.'.format(score))
regr_tree = DecisionTreeRegressor(max_depth=12)

regr_tree.fit(X_train,y_train)

results = pd.DataFrame(index=X_train.columns, data={'importance':regr_tree.feature_importances_})

score = regr_tree.score(X_train,y_train)

print('The r2 score is {:5.3f}.'.format(score))
test.dropna(inplace=True)

relevant = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X_test = test[relevant].copy()

dum_sex = pd.get_dummies(train.Sex).copy()

dum_emb = pd.get_dummies(train.Embarked).copy()



#dum_sex.head()

#dum_emb.head()

X_test.drop(['Sex','Embarked'], inplace=True, axis=1)

X_test = pd.concat([X_test, dum_sex, dum_emb], axis=1).copy()
from sklearn.tree import DecisionTreeClassifier

X_test.dropna(inplace=True)

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

dtc.predict(X_test)

acc_decision_tree = round(dtc.score(X_train, y_train) * 100, 2)

print (acc_decision_tree)