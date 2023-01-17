# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train.describe()

X = train
y = X["Survived"]
X=X.drop('Survived')
X.drop('Survived',axis=1,inplace=True)
X_test = test

X_test
X.Age.fillna(X.Age.mean(),inplace=True)
X.describe()
X_test.Age.fillna(X_test.Age.mean(),inplace=True)
X_test.describe()
X.dtypes
numeric = list(X.dtypes[X.dtypes!='object'].index)

model = rf(n_estimators = 100, oob_score = True, random_state=42)
model.fit(X[numeric],y)
model.oob_score_
y_oob = model.oob_prediction_
roc_auc_score(y,y_oob)
def clean(x):
    try:
        return x[0]
    except TypeError:
        return "None"
X.Cabin = X.Cabin.apply(clean)
        
X_test.Cabin = X_test.Cabin.apply(clean)
X.drop("Name",inplace = True,axis = 1)
X.drop("Ticket",axis=1,inplace = True)
X.drop("PassengerId",axis=1,inplace=True)
X

X.describe()
category = ['Sex','Cabin','Embarked']
for cat in category:
    X[cat].fillna('Missing',inplace=True)
    dumm = pd.get_dummies(X[cat],prefix=cat)
    X = pd.concat([X,dumm],axis=1)
    X.drop(cat,axis=1,inplace=True)
X
model = rf(100,oob_score = True, n_jobs=-1,random_state = 42)
model.fit(X,y)
roc_auc_score(y,model.oob_prediction_)
model2 = rf(1000)
model2.fit(X,y)
import matplotlib.pyplot as plt
fimp = model.feature_importances_
fimpp = pd.Series(fimp,index=X.columns)
fimpp = fimpp.sort_values()
fimpp.plot(kind="barh",figsize=(7,6))
model = rf(1000,oob_score = True, n_jobs = -1, min_samples_leaf=5,random_state = 42)
model.fit(X,y)
roc_auc_score(y,model.oob_prediction_)
X_test.Fare.fillna(X_test.Fare.mean(),inplace=True)
X_test

take = pd.read_csv('../input/test.csv').PassengerId
X_test.drop('PassengerId',inplace=True,axis = 1)
X_test
X_test.drop('Name',inplace=True,axis=1)
X_test
X_test
X_test.drop('Ticket',inplace=True,axis=1)
X_test
X_test
for cat in category:
    X_test[cat].fillna('Missing',inplace = True)
    dumm = pd.get_dummies(X_test[cat],prefix = cat)
    X_test = pd.concat([X_test,dumm],axis=1)
    X_test.drop(cat,axis=1,inplace=True)
X_test.columns
X.drop('Cabin_T',axis=1,inplace=True)
X.drop('Embarked_Missing',axis=1,inplace=True)
model = rf(100)
model.fit(X,y)

roc_auc_score(y,model.oob_prediction_)
y_test = model.predict(X_test)
y_test

#
output = pd.DataFrame({'PassengerId':take,'Survived':y_test})
#output.Survived = output.Survived.apply(lambda x: 1.0 if x>=0.5 else 0.0)
output.Survived = output.Survived.astype(int)
output.to_csv('output.csv', index=False)
