import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv',index_col=0)

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv',index_col=0)

print(train.head())
train = train.fillna(0)

test = test.fillna(0)

plt.figure(figsize=(15,15))

sns.heatmap(train.corr())
cols = ['GarageCars','GarageArea','GrLivArea','OverallQual']
f, axes = plt.subplots(1, 4,figsize=(20,10))

for i,col in enumerate(cols):

    sns.scatterplot(x=col, y="SalePrice", data=train,ax=axes[i])
train = train.drop(train[(train['OverallQual'] == 10) & (train['SalePrice'] < 12.5)].index)
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.model_selection import train_test_split

X = train[cols]

sc = StandardScaler()

sc.fit(X[['GarageArea','GrLivArea']].values)

X[['GarageArea','GrLivArea']] = sc.transform(X[['GarageArea','GrLivArea']].values) 

y = train['SalePrice']

s = y.std()

mean = y.mean()

y = (y - mean) / s

X_train, X_test, y_train, y_test = train_test_split(X,y)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X_train,y_train)
model.score(X_test,y_test)
X = test[cols]

X[['GarageArea','GrLivArea']] = sc.transform(X[['GarageArea','GrLivArea']].values) 

y_pred = model.predict(X)

pred = (y_pred * s) + mean 

sub = pd.DataFrame()

sub['Id'] = test.index.values

sub['SalePrice'] = pred

print(sub.head())

sub.to_csv('pred_submission.csv', index=False, encoding='utf-8')