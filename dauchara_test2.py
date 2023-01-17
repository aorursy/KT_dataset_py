# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import xgboost

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Train and Test Data

df_all = train.append(test)



print(test.shape, train.shape, df_all.shape)
## Null Counting Fucntion

def null_values(df):

    

    sum_null = df.isnull().sum()

    total = df.isnull().count()

    percent_nullvalues = 100* sum_null / total 

    df_null = pd.DataFrame()

    df_null['Total'] = total

    df_null['Null_Count'] = sum_null

    df_null['Percent'] = round(percent_nullvalues,2)

    df_null = df_null.sort_values(by='Null_Count',ascending = False)

    df_null = df_null[df_null.Null_Count > 0]

    

    return(df_null)

null_values(df_all)
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType'):

    df_all[col] = df_all[col].fillna('None')

    

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType'):

    df_all[col] = df_all[col].fillna('None')

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'):

    df_all[col] = df_all[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_all[col] = df_all[col].fillna('None')    



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_all[col] = df_all[col].fillna(0)



# Total area is the most important in terms of prices.    

df_all['TotalSF'] = df_all['TotalBsmtSF'] + df_all['1stFlrSF'] + df_all['2ndFlrSF']
null_values(df_all)
df_all.Electrical.value_counts()
# set mode as a representative value of electric

df_all['Electrical'] = df_all['Electrical'].fillna('SBrkr')
df_all['LotFrontage'] = df_all.groupby('BldgType')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
# remove Utilities, cause its only one in the train set

df_all['Utilities'] = df_all.drop('Utilities', axis=1) 
# year and month turn into categorical features

df_all.YrSold = df_all.YrSold.astype(str)

df_all.MoSold = df_all.MoSold.astype(str)
train = df_all[df_all.Id < 1461]

test = df_all[df_all.Id >= 1461].drop('SalePrice', axis=1)
null_values(train)

null_values(test)
# put the mode each column

test['Exterior1st'] = test['Exterior1st'].fillna('VinylSd')

test['Exterior2nd'] = test['Exterior2nd'].fillna('VinylSd')

test['KitchenQual'] = test['KitchenQual'].fillna('TA')

test['Functional'] = test['Functional'].fillna('Typ')

test['MSZoning'] = test['MSZoning'].fillna('RL')

test['SaleType'] = test['SaleType'].fillna('WD')
cols = ( 

        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'Electrical',

        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType',

        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

        'ExterQual', 'ExterCond','HeatingQC','KitchenQual', 'Functional', 'MSZoning', 'LandContour',

        'LandSlope', 'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'LotConfig',

        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',

        'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'SaleType', 'SaleCondition',

        'MSSubClass', 'OverallCond', 'YrSold', 'MoSold'

        )



for i in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(train[i].values)) 

    train[i] = lbl.transform(list(train[i].values))

    

    lbl.fit(list(test[i].values)) 

    test[i] = lbl.transform(list(test[i].values))
train['SalePriceLog'] = np.log(train.SalePrice)



corr = train.corr().abs()

corr.SalePriceLog[corr.SalePriceLog >= 0.5].sort_values(ascending=False)
plt.subplot2grid((2,1),(0,0))

train.SalePrice.plot(kind='kde')

plt.title('SalePrice')



plt.subplot2grid((2,1),(1,0))

train.SalePriceLog.plot(kind='kde')

plt.title('SalePriceLog')



plt.subplots_adjust(hspace = 0.5, wspace = 0.3)

plt.show()
train.plot.scatter(x='OverallQual', y='SalePriceLog')

plt.show()
train.plot.scatter(x='TotalSF', y='SalePriceLog')

plt.show()
# reset the outliers to the SalePriceLog at 95% confidence intervals below

for i in np.arange(10):

    pno = train.SalePriceLog[train.OverallQual == i+1]

    n = len(pno)

    mean = pno.mean()

    std = pno.std()

    upper_interval = mean + 1.96 * std

    lower_interval = mean - 1.96 * std

    train = train.drop(train[(train.OverallQual == i+1) & (train.SalePriceLog > upper_interval)].index)

    train = train.drop(train[(train.OverallQual == i+1) & (train.SalePriceLog < lower_interval)].index)



print(train.shape)
train.plot.scatter(x='OverallQual', y='SalePriceLog')

plt.show()
train.plot.scatter(x='TotalSF', y='SalePriceLog')

plt.show()
y_train = train.SalePriceLog.values.reshape((1370,1))

x_train = train.drop(['SalePrice','SalePriceLog'], axis=1).values.reshape((1370, 81))

x_test = test.values.reshape((1459,81))
xgb = xgboost.XGBRegressor(colsample_bytree=0.8, subsample=0.5,

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.8, n_estimators=2000,

                             reg_alpha=0.1, reg_lambda=0.3, gamma=0.01, 

                             silent=1, random_state =7, nthread = -1)





xgb.fit(x_train, y_train)

xgb_pred = xgb.predict(x_test)

y_test = xgb.predict(x_train)
RMSE = np.sqrt(mean_squared_error(y_train, y_test))

print(RMSE.round(4))
submission = pd.DataFrame({

        "Id": test['Id'],

        "SalePrice": np.exp(xgb_pred)

    })
submission.to_csv('submission.csv', index=False)