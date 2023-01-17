# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import sys

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from pandas.plotting import scatter_matrix

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score

import seaborn as sns

import mpl_toolkits

import matplotlib.pyplot as plt

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (10, 6)



%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head()
print ("Skew is:", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()
target = np.log(train.SalePrice)

print ("Skew is:", target.skew())

plt.hist(target, color='blue')

plt.show()
# Working with Numeric Features



numeric_features = train.select_dtypes(include=[np.number])

numeric_features.dtypes
train.OverallQual.unique()

quality_pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
plt.scatter(x=train['GrLivArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Above grade (ground) living area square feet')

plt.show()
plt.scatter(x=train['GarageArea'], y=target)

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
train = train[train['GarageArea'] < 1200]

plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))

plt.xlim(-200,1600) # This forces the same scale as before

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
# Handling Null Values

nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])

nulls.columns = ['Null Count']

nulls.index.name = 'Feature'

nulls
print ("Unique values are:", train.MiscFeature.unique())

# working with the non-numeric Features



categoricals = train.select_dtypes(exclude=[np.number])

categoricals.describe()

print ("Original: \n")

print (train.Street.value_counts(), "\n")
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)

test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print ('Encoded: \n')

print (train.enc_street.value_counts())
condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
def encode(x):

 return 1 if x == 'Partial' else 0

train['enc_condition'] = train.SaleCondition.apply(encode)

test['enc_condition'] = test.SaleCondition.apply(encode)
condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)

condition_pivot.plot(kind='bar', color='blue')

plt.xlabel('Encoded Sale Condition')

plt.ylabel('Median Sale Price')

plt.xticks(rotation=0)

plt.show()
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

sum(data.isnull().sum() != 0)

# Build a linear model

y = np.log(train.SalePrice)

X = data.drop(['SalePrice', 'Id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
lr = LinearRegression()

model = lr.fit(X_train, y_train)

print ("R^2 is: \n", model.score(X_test, y_test))

predictions = model.predict(X_test)

print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test

plt.scatter(predictions, actual_values, alpha=.7,

            color='b') #alpha helps to show overlapping data

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Linear Regression Model')

plt.show()
# Try to improve the model

for i in range (-2, 3):

    alpha = 10**i

    rm = linear_model.Ridge(alpha=alpha)

    ridge_model = rm.fit(X_train, y_train)

    preds_ridge = ridge_model.predict(X_test)



    plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')

    plt.xlabel('Predicted Price')

    plt.ylabel('Actual Price')

    plt.title('Ridge Regularization with alpha = {}'.format(alpha))

    overlay = 'R^2 is: {}\nRMSE is: {}'.format(

                    ridge_model.score(X_test, y_test),

                    mean_squared_error(y_test, preds_ridge))

    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

    plt.show()
submission = pd.DataFrame()

submission['Id'] = test.Id





feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()

subpredictions = model.predict(feats)



#reverse to exp from log predict

final_predictions = np.exp(subpredictions)

submission['SalePrice'] = final_predictions
