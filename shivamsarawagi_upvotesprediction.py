import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import ensemble

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import SGDClassifier

from sklearn import preprocessing
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
test = pd.read_csv('../input/test_8i3B3FC.csv')

train = pd.read_csv('../input/train_NIR5Yl1.csv')
train.head()
train.isnull().sum()
test.head()
test.isnull().sum()
train.shape
test.shape
unique, counts = np.unique(train['Tag'], return_counts=True)

print(unique)

print(counts)
unique, counts = np.unique(test['Tag'], return_counts=True)

print(unique)

print(counts)
yTrain=train['Upvotes']

train.drop(labels=['Upvotes','ID','Username'],axis=1,inplace=True)
ID_test = test['ID']

test.drop(labels=['ID','Username'],axis=1,inplace=True)

train.head()
test = pd.get_dummies(test)
train = pd.get_dummies(train)
train.head()
test.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train[['Reputation','Views','Answers']] = scaler.fit_transform(train[['Reputation','Views','Answers']])

test[['Reputation','Views','Answers']] = scaler.fit_transform(test[['Reputation','Views','Answers']])

train.head(5)
from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(train,

yTrain,test_size=0.20, random_state=42)
from sklearn.linear_model import SGDRegressor



reg = SGDRegressor(loss = 'epsilon_insensitive', verbose=1,eta0=0.1,n_iter=60)



reg.fit(X_train, Y_train)
model = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 100, verbose=1)

model.fit(X_train, Y_train)
from sklearn.metrics import mean_squared_error

Y_pred = model.predict(X_valid)

error = mean_squared_error(Y_valid, Y_pred)

print(error)
model.score(X_train, Y_train)
Upvotes = pd.Series(np.abs(model.predict(test)), name="Upvotes")



results = pd.concat([ID_test,Upvotes],axis=1)



results.to_csv("submission.csv",index=False)