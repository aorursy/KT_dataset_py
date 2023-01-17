# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation

%matplotlib inline 

sns.set(color_codes=True)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.shape
train.shape
n = train.isnull().sum().sort_values(ascending=False)

train_nan = (n[n>0])

dict(train_nan)

train_nan.head(50)
n_test = test.isnull().sum().sort_values(ascending=False)

test_nan = (n[n>0])

dict(train_nan)

train_nan.head(50)
train = train.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu','GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageArea'],axis = 1)
test = test.drop(['Alley','PoolQC','Fence','MiscFeature','FireplaceQu','GarageYrBlt','TotRmsAbvGrd','1stFlrSF','GarageArea'],axis = 1)
Y = train.iloc[:,-1].values
train = train.drop(['SalePrice'],axis = 1)
train_num = train._get_numeric_data()

train_cat =train.select_dtypes(exclude = ['number'])
test_num = test._get_numeric_data()

test_cat = test.select_dtypes(exclude = ['number'])
col = test_num.columns.tolist()

col2 = test_cat.columns.tolist()
test[col2] = test[col2].fillna("None")

test[col] = test[col].fillna(0)
cols = train_num.columns.tolist()

cols2 = train_cat.columns.tolist()
train[cols] = train[cols].fillna(0)

train[cols2] = train[cols2].fillna("None")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX = LabelEncoder()
for i in train_cat.columns:

    train[i] = labelEncoderX.fit_transform(train[i])
for i in test_cat.columns:

    test[i] = labelEncoderX.fit_transform(test[i])
#onehotEncoder = OneHotEncoder()
#train = onehotEncoder.fit_transform(train)
#train = pd.concat([train_num, train_cat.reindex(train_num.index)], axis=1)
X = train
train.head()
test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
from sklearn.metrics import mean_squared_error



from sklearn import ensemble



from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()



linear_regressor.fit(X_train, y_train)



y_hat =  linear_regressor.predict(X_train)



mse = mean_squared_error(y_train, y_hat)

 

print('mse: ', mse)
import math
rmse = math.sqrt(mse)

print('rmse ', rmse)
params = {

    'n_estimators': 1,

    'max_depth': 1,

    'learning_rate': 1,

    'criterion': 'mse'

}



gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)



gradient_boosting_regressor.fit(X_train, y_train)



y_hat2 =  gradient_boosting_regressor.predict(X_train)

mse = mean_squared_error(y_train, y_hat2)

 

print('mse: ', mse)
rmse = math.sqrt(mse)

print('rmse ', rmse)
params['n_estimators'] = 2



gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)



gradient_boosting_regressor.fit(X_train, y_train)



y_hat2 = gradient_boosting_regressor.predict(X_train)

y_hat2.shape
mse = mean_squared_error(y_train, y_hat2)

 

print('mse: ', mse)
rmse = math.sqrt(mse)

print('rmse ', rmse)
for idx, n_estimators in enumerate([5, 10, 20, 50]):

    params['n_estimators'] = n_estimators



    gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)



    gradient_boosting_regressor.fit(X_train, y_train)

    

    y_hat3 = gradient_boosting_regressor.predict(X_train)
mse3 = mean_squared_error(y_train, y_hat3)

 

print('mse: ', mse3)
rmse3 = math.sqrt(mse3)

print('rmse ', rmse3)
params['n_estimators'] = 10

for idx, max_depth in enumerate([1, 2, 3, 5]):

    params['max_depth'] = max_depth



    gradient_boosting_regressor = ensemble.GradientBoostingRegressor(**params)

    gradient_boosting_regressor.fit(X_train, y_train)

    y_hatt = gradient_boosting_regressor.predict(X_train)
mset = mean_squared_error(y_train, y_hatt)

 

print('mse: ', mset)
rmset = math.sqrt(mset)

print('rmse ', rmset)
gradient_boosting_regressor.fit(X_test, y_test)

y_hat_test = gradient_boosting_regressor.predict(X_test)
mset = mean_squared_error(y_test, y_hat_test)

 

print('mse: ', mset)
rmset = math.sqrt(mset)

print('rmse ', rmset)
yy = gradient_boosting_regressor.predict(test)
mse = mean_squared_error(y_test, y_hat_test)

 

print('mse: ', mset)
yy.shape
test.shape
submission.shape
sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
sales=pd.DataFrame(yy,columns=['SalePrice'])

sample_submission['SalePrice']=sales['SalePrice']

sample_submission.head()
sample_submission.head()
rr = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
s = pd.DataFrame()

s['Id'] = rr['Id']

s['SalePrice'] = pd.DataFrame(yy,columns=['SalePrice'])

s.to_csv('MySubmission.csv',index=False)
a = pd.read_csv('MySubmission.csv')
a.isnull().sum()