import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.describe(include='O')
df_train.describe()
df_train.info()
df_train = df_train[df_train['LotArea']<15000]
sns.scatterplot(x='MiscFeature', y='SalePrice', data=df_train)

#maioria é shred, drop
sns.scatterplot(x='Alley', y='SalePrice', data=df_train)

#poucas variaveis drop
sns.scatterplot(x='PoolQC', y='SalePrice', data=df_train)

#poucas variaveis, drop
sns.scatterplot(x='PoolArea', y='SalePrice', data=df_train)

#poucas variaveis, drop
sns.scatterplot(x='BsmtCond', y='SalePrice', data=df_train)

#poucas variaveis, drop
df_train['LandSlope'].value_counts()

sns.scatterplot(x='LandSlope', y='SalePrice', data=df_train)

#drop não tem muita variação
df_train['Neighborhood'].value_counts()

sns.scatterplot(x='Neighborhood', y='SalePrice', data=df_train)
df_train['PoolArea'].value_counts()

#drop não serve pra nada
df_train['ScreenPorch'].value_counts()
df_train['Fence'].value_counts()
df_train['SaleCondition'].value_counts()

df_train['LandContour'].value_counts()

df_train['Utilities'].value_counts()

#drop maioria allpub
df_train['LotConfig'].value_counts()
drop_columns = ['PoolQC','Alley',  'MiscFeature', 'MasVnrArea', 'Fence', 'PoolArea', 'MiscVal', 'FireplaceQu', 'GarageFinish', 'GarageYrBlt', 'Condition1', 'Condition2', 'Utilities', 'LandSlope', 'Street', 'RoofMatl','Exterior1st', 'Exterior2nd', 'PavedDrive', 'Functional', 'Electrical', 'CentralAir', 'Heating', 'RoofStyle', 'LandContour']
for coluna in drop_columns:

    df_train.drop([coluna], axis=1, inplace=True)
df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean(), inplace=True)
df_train['BsmtQual'].fillna('TA', inplace=True)
df_train['BsmtCond'].fillna('TA', inplace=True)
df_train['BsmtExposure'].fillna('No', inplace=True)
df_train['BsmtFinType2'].fillna('Unf', inplace=True)
df_train['LotFrontage'].fillna(df_train['LotFrontage'].median(),inplace=True)
df_train['GarageQual'].fillna('TA', inplace=True)
df_train['GarageType'].fillna('TA', inplace=True)
df_train['MasVnrType'].fillna('None', inplace=True)
df_train['BsmtFinType1'].fillna('Unf', inplace=True)
df_train.replace({'TA':0}, inplace=True)

df_train.replace({'Gd':1}, inplace=True)

df_train.replace({'Ex':2}, inplace=True)

df_train.replace({'Fa':3}, inplace=True)

df_train.replace({'Po':4}, inplace=True)
dumm = ['MSZoning','LotShape', 'LotConfig', 'Neighborhood', 'BldgType','HouseStyle', 'MasVnrType',  'Foundation','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'SaleType', 'SaleCondition']
for variavel in dumm:

    df_dumm = pd.get_dummies(df_train[variavel])

    df_train = pd.concat([df_train, df_dumm], axis=1)
df_train.reset_index(inplace=True)
df_train.drop(['index'], axis=1, inplace=True)
for coluna in dumm:

    df_train.drop([coluna], axis=1, inplace=True)
df_train.fillna(0,inplace=True)
X = df_train.drop(['SalePrice'], axis=1)

Y= df_train['SalePrice']
X.drop(['2.5Fin'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
import numpy as np
X_train, x_test, Y_train, y_test =  train_test_split(X, Y, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor

from sklearn.model_selection import RandomizedSearchCV
# dtc = RandomForestRegressor()
# dtc.fit(X_train, Y_train)
params = {'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# rf_random = RandomizedSearchCV(estimator = dtc, param_distributions = params, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# rf_random.fit(X_train, Y_train)
dtc = LGBMRegressor(n_estimators=800, min_samples_split=2)
# dtc = RandomForestRegressor(n_estimators=800,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=70,bootstrap=False)
dtc.fit(X_train, Y_train)
# df_fi = pd.DataFrame()
# df_fi['Feature'] = X_train.columns
# df_fi['Importance'] = dtc.feature_importances_
# df_fi.to_excel('feature_importance.xlsx')
y_pred = dtc.predict(x_test)

y_pred_train = dtc.predict(X_train)
from sklearn import metrics

print('RMSLE treinamento', metrics.mean_squared_log_error(Y_train, y_pred_train))

print('MAE treinamento:', metrics.mean_absolute_error(Y_train, y_pred_train))

print('MSE treinamento:', metrics.mean_squared_error(Y_train, y_pred_train))

print('RMSE treinamento:', np.sqrt(metrics.mean_squared_error(Y_train, y_pred_train)))

print('\n')



print('MAE teste:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE teste:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE teste:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('RMSLE teste', metrics.mean_squared_log_error(y_test, y_pred))
print('RMSLE', metrics.mean_squared_log_error(Y_train, y_pred_train))
drop_columns = ['PoolQC','Alley',  'MiscFeature', 'MasVnrArea', 'Fence', 'PoolArea', 'MiscVal', 'FireplaceQu', 'GarageFinish', 'GarageYrBlt', 'Condition1', 'Condition2', 'Utilities', 'LandSlope', 'Street', 'RoofMatl','Exterior1st', 'Exterior2nd', 'PavedDrive', 'Functional', 'Electrical', 'CentralAir', 'Heating', 'RoofStyle', 'LandContour']
for coluna in drop_columns:

    df_test.drop([coluna], axis=1, inplace=True)
df_test['LotFrontage'].fillna(df_train['LotFrontage'].mean(), inplace=True)
df_test['BsmtQual'].fillna('TA', inplace=True)
df_test['BsmtCond'].fillna('TA', inplace=True)
df_test['BsmtExposure'].fillna('No', inplace=True)
df_test['BsmtFinType2'].fillna('Unf', inplace=True)
df_test['LotFrontage'].fillna(df_train['LotFrontage'].median(),inplace=True)
df_test['GarageQual'].fillna('TA', inplace=True)
df_test['GarageType'].fillna('TA', inplace=True)
df_test['MasVnrType'].fillna('None', inplace=True)
df_test['BsmtFinType1'].fillna('Unf', inplace=True)
df_test.replace({'TA':0}, inplace=True)

df_test.replace({'Gd':1}, inplace=True)

df_test.replace({'Ex':2}, inplace=True)

df_test.replace({'Fa':3}, inplace=True)

df_test.replace({'Po':4}, inplace=True)
dumm = ['MSZoning','LotShape', 'LotConfig', 'Neighborhood', 'BldgType','HouseStyle', 'MasVnrType',  'Foundation','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'SaleType', 'SaleCondition']
for variavel in dumm:

    df_dumm = pd.get_dummies(df_test[variavel])

    df_test = pd.concat([df_test, df_dumm], axis=1)
df_test.reset_index(inplace=True)
df_test.drop(['index'], axis=1, inplace=True)
for coluna in dumm:

    df_test.drop([coluna], axis=1, inplace=True)
dtc.fit(X, Y)
df_test.fillna(0, inplace=True)
y_pred = dtc.predict(df_test)
df_sub = pd.DataFrame()

df_sub['Id'] = list(df_test['Id'])

df_sub['SalePrice'] = y_pred
df_sub.to_csv('answer.csv', index=False)