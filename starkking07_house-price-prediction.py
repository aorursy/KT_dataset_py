# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
from __future__ import absolute_import, division, print_function

import pathlib
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
training_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
training_data['SalePrice'].describe()
training_data['SalePrice'].skew()
training_data.info()
training_data['Skewed_SP'] = np.log(training_data['SalePrice']+1)
training_data['Skewed_SP'].skew()
plt.hist(training_data['Skewed_SP'], color='blue')
plt.show()
sns.factorplot('MSSubClass', 'Skewed_SP', data=training_data,kind='bar',size=5,aspect=3)
fig, (axis1) = plt.subplots(1,1,figsize=(10,3))
sns.countplot('MSSubClass', data=training_data)
training_data['MSSubClass'].value_counts()
sns.factorplot('MSZoning', 'Skewed_SP', data=training_data,kind='bar',size=3,aspect=3)
fig, (axis1) = plt.subplots(1,1,figsize=(10,3))
sns.countplot(x='MSZoning', data=training_data, ax=axis1)
training_data['MSZoning'].value_counts()
sns.factorplot(x='MSZoning', y='SalePrice', col='MSSubClass', data=training_data, kind='bar', col_wrap=4, aspect=0.8)
numerical_features = training_data.select_dtypes(include=[np.number])
numerical_features.dtypes
corr = numerical_features.corr()
#print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
#print (corr['SalePrice'].sort_values(ascending=False)[-5:])
print (corr['SalePrice'].sort_values(ascending=False)[:], '\n')
#Creating a pivot table 
quality_pivot = training_data.pivot_table(index='OverallQual',values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar',color='green')
plt.xlabel('Overall Quality')
plt.ylabel('Median')
plt.xticks(rotation=0)
plt.show()
sns.regplot(x='GrLivArea',y='Skewed_SP',data=training_data)
#Removing outliers
training_data = training_data[training_data['GrLivArea'] < 4000]
sns.regplot(x='GrLivArea',y='Skewed_SP',data=training_data)
sns.regplot(x='GarageArea', y='Skewed_SP', data = training_data)
#removing outliers
training_data = training_data[training_data['GarageArea']<1200]
sns.regplot(x='GarageArea', y='Skewed_SP', data = training_data)
null_values = pd.DataFrame(training_data.isnull().sum().sort_values(ascending=False)[:25])
null_values.columns = ['Null Value Count']
null_values.index.name = 'Feature'
null_values
categ_features = training_data.select_dtypes(exclude=[np.number])
categ_features.describe(include='all')
g = sns.factorplot(x='Condition1', y='Skewed_SP', col='Condition2', data=training_data, kind='bar', col_wrap=4, aspect=0.8)
g.set_xticklabels(rotation=90)
g = sns.factorplot(x='SaleCondition', y='Skewed_SP', col='SaleType', data=training_data, kind='bar', col_wrap=4, aspect=0.8)
g.set_xticklabels(rotation=90)
#Convert categorical variable into dummy/indicator variables
training_data['enc_street'] = pd.get_dummies(training_data.Street, drop_first=True)
test_data['enc_street'] = pd.get_dummies(training_data.Street, drop_first=True)
#Feature Engineering
condition_pivot = training_data.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

def encode(x): return 1 if x=='Partial' else 0
training_data['enc_condition'] = training_data.SaleCondition.apply(encode)
test_data['enc_condition'] = test_data.SaleCondition.apply(encode)
condition_pivot = training_data.pivot_table(index='enc_condition', values='SalePrice',
                                            aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
data = training_data.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
y = np.log(training_data.SalePrice)
X = data.drop(['SalePrice','Skewed_SP','Id'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
from sklearn import ensemble

#lr =  ensemble.RandomForestRegressor(n_estimators = 100, oob_score = True, n_jobs = -1,random_state =50,max_features = "sqrt", min_samples_leaf = 50)
#lr = linear_model.LinearRegression()
lr = ensemble.GradientBoostingRegressor()
#lr = linear_model.TheilSenRegressor()
#lr = linear_model.RANSACRegressor(random_state=50)
model = lr.fit(X_train, y_train)

model.score(X_test, y_test)
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
#pltrandom_state=None.show()

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
submission['Id'] = test_data.Id
feats = test_data.select_dtypes(
        include=[np.number]).drop('Id',axis=1).interpolate()
#predictions = model.predict(feats)
rm = linear_model.Ridge(alpha=100)
ridge_model = rm.fit(X_train, y_train)
predictions = ridge_model.predict(feats)
final_predictions = np.exp(predictions)
submission['SalePrice'] = final_predictions
submission.head()
submission.to_csv('submission.csv', index=False)


