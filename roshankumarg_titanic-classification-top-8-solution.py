# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/titanic/test.csv")
test.head()
print(train.info())
print("-------Missing values--------")
print(train.isnull().sum())
print("-------Missing percentage--------")
print(train.isnull().sum()/len(train))
col = ['Name','Ticket','Cabin']
train = train.drop(col,axis=1)
train.head(2)
#train = train.dropna()
train.columns
cols = ['Survived','Pclass','Sex','SibSp','Parch','Embarked']

for col in cols:
    print(col)
    print(train[col].value_counts())
dummies = []
cols = ['Sex','Embarked','Pclass']
for col in cols:
    dummies.append(pd.get_dummies(train[col]))
    
dummies_df = pd.concat(dummies,axis=1)
dummies_df.head()
train = pd.concat((train,dummies_df),axis=1)
train = train.drop(['Sex','Embarked','Pclass'],axis=1)
train.head(2)
train.isnull().sum()
train['Age'] = train['Age'].interpolate()
train.describe()
y = train['Survived']
X = train.drop(['Survived','PassengerId'],axis=1)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
model = DecisionTreeClassifier(max_depth = 5)
model.fit(X_train,y_train)
cv = cross_val_score(model,X_train,y_train,cv=10)
print(cv)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn import ensemble
m1 = ensemble.RandomForestClassifier(n_estimators=100)
m1.fit (X_train, y_train)
m1.score (X_test, y_test)

import lightgbm as lgb
m2 = lgb.LGBMClassifier(n_estimators=100)
m2.fit (X_train, y_train)
m2.score (X_test, y_test)
m3 = ensemble.GradientBoostingClassifier(n_estimators=50)
m3.fit (X_train, y_train)
m3.score (X_test, y_test)

"""
parameters = {

    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}
"""
"""
import lightgbm as lgb
from sklearn import metrics
model = lgb.LGBMClassifier(parameters,num_boost_round=5000,early_stopping_rounds=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))
#model.score(X_test, y_test)
"""

col = ['Name','Ticket','Cabin']
test = test.drop(col,axis=1)
dummies = []
cols = ['Sex','Embarked','Pclass']
for col in cols:
    dummies.append(pd.get_dummies(test[col]))
    
dummies_df = pd.concat(dummies,axis=1)
test = pd.concat((test,dummies_df),axis=1)
test = test.drop(['Sex','Embarked','Pclass'],axis=1)
test.head(2)
test['Age'] = test['Age'].interpolate()
test['Fare'] = test['Fare'] .fillna(test['Fare'].mean())
X_res = test.drop(['PassengerId'],axis=1)
y_pred = m3.predict(X_res)
sub = pd.DataFrame(test['PassengerId'])
sub['Survived'] = y_pred
sub.head()
sub.to_csv('titanic_results5.csv',index=False)