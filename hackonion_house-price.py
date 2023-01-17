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
import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.info()
train.columns
plt.figure(figsize=(10,6))

sns.distplot(train['SalePrice'], color='g')

plt.legend(['Normal dist.'])

plt.title('Distribution of Sales Price', fontsize=18)

plt.show() 
plt.figure(figsize=(30,10))

sns.heatmap(train.corr(),cmap='Greens',annot=True)

plt.show()
corr = train.corr()

top_corr_features = corr.index[abs(corr['SalePrice'])>0.5]

top_corr_features
plt.figure(figsize=(30,10))

sns.heatmap(train[top_corr_features].corr(),cmap='Greens',annot=True)

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='SalePrice',y='1stFlrSF',data=train , color="#DF3A41", alpha=0.6)

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='SalePrice',y='OverallQual', data=train)

plt.title('SalePrice and OverallQual')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='YearBuilt',y='SalePrice', data=train,color='c', alpha=0.6)

plt.title('YearBuilt and SalePrice')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='YearBuilt',y='SalePrice', data=train,color='g', alpha=0.6)

plt.title('YearBuilt and SalePrice')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x= 'SalePrice', y='GrLivArea',color='#DA43E1', data=train, alpha=0.6)

plt.show()
train = train[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',

       'SalePrice']]

test = test[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

       'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]
total_train_nan = train.isnull().sum().sort_values(ascending=False)

miss_train_data = pd.concat([total_train_nan], axis=1, keys=['Total'])

miss_train_data.head(15)
total_nan = test.isnull().sum().sort_values(ascending=False)

missi_data = pd.concat([total_nan], axis=1, keys=['Total'])

missi_data.head(15)
test['GarageArea'] = test['GarageArea'].fillna(train['GarageArea'].mean())

test['GarageCars'] = test['GarageCars'].fillna(train['GarageCars'].mean())

test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(train['TotalBsmtSF'].mean())
total_test_nan = test.isnull().sum().sort_values(ascending=False)

miss_test_data = pd.concat([total_test_nan], axis=1, keys=['Total'])

miss_test_data.head(15)
categorical_train = train.dtypes==object

categorical_train
categorical_test = test.dtypes==object

categorical_test
X_train, X_test, y_train, y_test = train_test_split(train.drop('SalePrice', axis=1), train['SalePrice'], test_size=0.2, random_state=1)



y_train= y_train.values.reshape(-1,1)

y_test= y_test.values.reshape(-1,1)



std_X = StandardScaler()

std_Y = StandardScaler()

X_train = std_X.fit_transform(X_train)

X_test = std_X.fit_transform(X_test)

y_train = std_Y.fit_transform(y_train)

y_test = std_Y.fit_transform(y_test)



tree = RandomForestRegressor(n_estimators=500 ,random_state=0)

tree.fit(X_train,y_train.ravel())



pred = tree.predict(X_test)



print('MSE:', metrics.mean_squared_error(y_test, pred))

print('Score train', tree.score(X_train,y_train))

print('Score test', tree.score(X_test,y_test))
take_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

Id = take_id['Id']

take_id = pd.DataFrame(Id, columns=['Id'])

test.shape
test = std_X.fit_transform(test)
test_pred = tree.predict(test)

test_pred= test_pred.reshape(-1,1)

test_pred.shape
test_pred_tree =std_Y.inverse_transform(test_pred)

test_pred_tree = pd.DataFrame(test_pred_tree, columns=['SalePrice'])
test_pred_tree.head()
result = pd.concat([take_id,test_pred_tree], axis=1)

result.head()
result.to_csv('submission.csv',index=False)