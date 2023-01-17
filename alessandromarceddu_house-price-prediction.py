

import os

import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor 

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_test.head()
corr_train = df_train.corr()

plt.subplots(figsize=(20,15))

sns.heatmap(corr_train, vmax=0.9, square=True, center = 0, cmap = 'RdGy')
features = ['Id','OverallQual', 'LotArea','YearBuilt', 'YearRemodAdd','MasVnrArea','TotalBsmtSF','1stFlrSF',

'GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','SalePrice']

train = df_train[features]
missing_data_train = train.isnull()



for column in missing_data_train.columns:

    print(column)

    print(missing_data_train[column].value_counts())

    print('')
sns.distplot(train['MasVnrArea'])
train['MasVnrArea'].fillna(0, inplace=True)
sns.distplot(train['GarageYrBlt'])
train['GarageYrBlt'].mode
train['GarageYrBlt'].fillna(2003, inplace=True)


ax = sns.regplot(x='OverallQual', y='SalePrice', data=train)


ax = sns.regplot(x='LotArea', y='SalePrice', data=train)


ax = sns.regplot(x='YearBuilt', y='SalePrice', data=train)


ax = sns.regplot(x='YearRemodAdd', y='SalePrice', data=train)


ax = sns.regplot(x='MasVnrArea', y='SalePrice', data=train)


ax = sns.regplot(x='TotalBsmtSF', y='SalePrice', data=train)


ax = sns.regplot(x='1stFlrSF', y='SalePrice', data=train)


ax = sns.regplot(x='GrLivArea', y='SalePrice', data=train)


ax = sns.regplot(x='FullBath', y='SalePrice', data=train)


ax = sns.regplot(x='TotRmsAbvGrd', y='SalePrice', data=train)


ax = sns.regplot(x='Fireplaces', y='SalePrice', data=train)


ax = sns.regplot(x='GarageYrBlt', y='SalePrice', data=train)


ax = sns.regplot(x='GarageCars', y='SalePrice', data=train)


ax = sns.regplot(x='GarageArea', y='SalePrice', data=train)
features = ['Id','OverallQual', 'YearBuilt','TotalBsmtSF','1stFlrSF',

'GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','SalePrice']

train= train[features]
x = ['Id','OverallQual', 'YearBuilt','TotalBsmtSF','1stFlrSF',

'GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea']

X = train[x]



Y=train['SalePrice']
x_train, x_test, y_train, y_test=train_test_split(X,Y, test_size=0.2)
LR = LinearRegression()

LR.fit(x_train,y_train)

yhat= LR.predict(x_test)

print('The accuracy of the model is:', r2_score(yhat,y_test))
DT = DecisionTreeRegressor()

DT.fit(x_train,y_train)

yhat1 = DT.predict(x_test)

print('The accuracy of the model is:', r2_score(yhat1,y_test))
RF = RandomForestRegressor()

RF.fit(x_train, y_train)

yhat2 = RF.predict(x_test)

print('The accuracy of the model is:', r2_score(yhat2,y_test))
XGB = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGB.fit(x_train, y_train)

yhat3 = XGB.predict(x_test)

print('The accuracy of the model is:',r2_score(yhat3,y_test))
features = ['Id','OverallQual', 'YearBuilt','TotalBsmtSF','1stFlrSF',

'GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea']

test = df_test[features]
model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model.fit(x_train, y_train)

predictions = model.predict(test)

output = pd.DataFrame({'Id': test.Id, 'SalePrice': predictions})

output.to_csv('sample_submission.csv', index=False)

print("Your submission was successfully saved!")