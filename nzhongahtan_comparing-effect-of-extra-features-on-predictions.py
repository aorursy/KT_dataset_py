import numpy as np

import pandas as pd

import os

from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv("../input/titanic/test.csv")



train_features = pd.read_csv('../input/data-w-features/train (1).csv')

test_features = pd.read_csv('../input/data-w-features/test (1).csv')
train.head()
train_features.head()
train['Sex'] = train['Sex'].astype('category').cat.codes

test['Sex'] = test['Sex'].astype('category').cat.codes

train['Embarked']  = train['Embarked'].astype('category').cat.codes

test['Embarked']  = test['Embarked'].astype('category').cat.codes
x = train.drop(['Survived','Name','Ticket','Cabin'],axis = 1)

y = train.Survived

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state = 242)

model = XGBClassifier()

model.fit(train_x,train_y)

preds = model.predict(val_x)

print(accuracy_score(val_y,preds))
train['Age'] = train['Age'].fillna(train['Age'].mean())

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode())

x = train.drop(['Survived','Name','Ticket','Cabin'],axis = 1)

y = train.Survived

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state = 242)

forest = RandomForestClassifier(random_state=2)

forest.fit(train_x,train_y)

forest_preds=forest.predict(val_x)

forest_score = accuracy_score(forest_preds,val_y)

print(forest_score)
neigh = KNeighborsClassifier (n_neighbors=5)

neigh.fit(train_x,train_y)

neigh_preds = neigh.predict(val_x)

#neigh_preds = np.round(neigh_preds,0,out=None)

#neigh_preds.astype(int)

neigh_score = accuracy_score(neigh_preds,val_y)

print(neigh_score)
logs = LogisticRegression()

logs.fit(train_x,train_y)

logs_preds = logs.predict(val_x)

logs_preds = np.round(logs_preds,0,out=None)

logs_preds.astype(int)

logs_score = accuracy_score(logs_preds,val_y)

print (logs_score)
x = train_features.drop(['Survived'],axis = 1)

y = train_features.Survived

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state = 242)
model = XGBClassifier()

model.fit(train_x,train_y)

preds = model.predict(val_x)

print(accuracy_score(val_y,preds))
forest = RandomForestClassifier(random_state=2)

forest.fit(train_x,train_y)

forest_preds=forest.predict(val_x)

forest_score = accuracy_score(forest_preds,val_y)

print(forest_score)
neigh = KNeighborsClassifier (n_neighbors=5)

neigh.fit(train_x,train_y)

neigh_preds = neigh.predict(val_x)

#neigh_preds = np.round(neigh_preds,0,out=None)

#neigh_preds.astype(int)

neigh_score = accuracy_score(neigh_preds,val_y)

print(neigh_score)
logs = LogisticRegression()

logs.fit(train_x,train_y)

logs_preds = logs.predict(val_x)

logs_preds = np.round(logs_preds,0,out=None)

logs_preds.astype(int)

logs_score = accuracy_score(logs_preds,val_y)

print (logs_score)