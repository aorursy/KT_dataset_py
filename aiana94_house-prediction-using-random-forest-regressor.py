# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 80)

pd.set_option('display.max_rows', 80)





from scipy import stats

from scipy.stats import norm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Visualization Libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





# Model building libraries

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import Imputer



from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(10)
train.columns
# Analysing the dependent variable:

train['SalePrice'].describe()
sns.distplot(train['SalePrice'] , fit=norm)
train['SalePrice'] = np.log(train['SalePrice'])

sns.distplot(train['SalePrice'] , fit=norm)
# Finding missing value % for each column



percent_missing = train.isnull().sum() * 100 / len(train)

missing_value_df = pd.DataFrame({'column_name': train.columns,

                                 'percent_missing': percent_missing})



missing_value_df.sort_values('percent_missing', ascending = False)
corr_mat = train.corr()



f, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(corr_mat, vmax=.8, square=True)
train.columns.get_loc('Fence')
# Dropping these columns: PoolQC(72), MiscFeature(74), Alley(6), Fence(73)  >80% (less correlated and missing value % is high)

train.drop(train.columns[[6,72,73,74]], axis=1, inplace=True)

test.drop(test.columns[[6,72,73,74]], axis=1, inplace=True)
# Overall Quality - rating from 1 to 10

f, ax = plt.subplots(figsize=(10, 8))

fig = sns.boxplot('OverallQual', y="SalePrice", data=train)

fig.axis(ymin=10, ymax=14);
train = train.drop(train[(train['OverallQual'] == 10)& (train['SalePrice'] <12.3)].index)
# TotalBsmtSF : Total square feet of basement area

fig, ax = plt.subplots()

ax.scatter(train['TotalBsmtSF'], train['SalePrice'])

plt.show()
# GrLivArea : Above grade (ground) living area square feet

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.show()
# GarageCars/GarageArea: both are highly correlated to each other. let's check only one



# GarageCars - Size of garage in car capacity from 0 to 4

f, ax = plt.subplots(figsize=(10, 8))

fig = sns.boxplot('GarageCars', y="SalePrice", data=train)

fig.axis(ymin=10, ymax=14);
# YearBuilt: Original construction date

f, ax = plt.subplots(figsize=(10, 8))

fig = sns.boxplot('YearBuilt', y="SalePrice", data=train)

fig.axis(ymin=10, ymax=14);
# FullBath: Full bathrooms above grade: from 0 to 3



f, ax = plt.subplots(figsize=(10, 8))

fig = sns.boxplot('FullBath', y="SalePrice", data=train)

fig.axis(ymin=10, ymax=14);
data = pd.concat((train, test)).reset_index(drop=True)
len(data)
train_rows = train.shape[0]

test_rows = test.shape[0]

percent_missing = data.isnull().sum() * 100 / len(data)



missing_value_df = pd.DataFrame({'column_name': data.columns,

                                 'percent_missing': percent_missing})



missing_value_df[missing_value_df['percent_missing']>0].sort_index()
data["FireplaceQu"] = data["FireplaceQu"].fillna("None")
data['LotFrontage'].describe()
data["LotFrontage"] = data["LotFrontage"].fillna(data['LotFrontage'].mean())
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[col] = data[col].fillna('None')
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:

    data[col] = data[col].fillna(0)
for col in ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'BsmtCond']:

    data[col] = data[col].fillna('None')
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:

    data[col] = data[col].fillna(0)
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['Exterior1st'].mode()[0]
data['Exterior2nd'].mode()[0]
for col in ['Exterior1st', 'Exterior2nd']:

    data[col] = data[col].fillna(data['Exterior2nd'].mode()[0])
data['Functional'].mode()[0]
data['Functional'] = data['Functional'].fillna(data['Functional'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
data = data.drop(['Utilities'], axis=1)
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
percent_missing = data.isnull().sum() * 100 / len(data)



missing_value_df = pd.DataFrame({'column_name': data.columns,

                                 'percent_missing': percent_missing})



missing_value_df[missing_value_df['percent_missing']>0].sort_index()
data.columns
# Converting few numerical variables to categorical

data['MSSubClass'] = data['MSSubClass'].apply(str)

data['OverallCond'] = data['OverallCond'].astype(str)







data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)

data['Heating'] = data['Heating'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC',  'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope','RoofStyle','SaleCondition',

        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 'RoofMatl', 'SaleType',

        'YrSold', 'MoSold', 'BldgType', 'Condition1', 'Condition2', 'Electrical', 'Exterior1st', 'Exterior2nd', 

        'Foundation', 'GarageType','Heating', 'HouseStyle', 'LotConfig', 'LandContour', 'MasVnrType', 'Neighborhood','MSZoning')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(data[c].values)) 

    data[c] = lbl.transform(list(data[c].values))

# shape        

print('Shape all_data: {}'.format(data.shape))
train = data[data['Id']<=1460]

test = data[data['Id']>1460]
y = train['SalePrice']

X = train.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
rfr = RandomForestRegressor(n_estimators = 100, random_state = 42)

rfr.fit(train_X, train_y);
predictions = rfr.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
test = test.drop(['SalePrice'], axis=1)
predictions_RF = rfr.predict(test)
my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)

test_Z = my_imputer.transform(test)
XGB = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

XGB.fit(train_X, train_y, verbose=False)
predictions = XGB.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
# Tuning XGBoost model

XGB1 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

XGB1.fit(train_X, train_y, early_stopping_rounds=5, 

             eval_set=[(test_X, test_y)], verbose=False)
predictions = XGB1.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
predictions_XGB = XGB1.predict(test_Z)
SaleP = pd.Series(np.exp(predictions_XGB))
test['SalePrice'] = SaleP


submission1 = test[['Id', 'SalePrice']]

submission1.to_csv('../submission1.csv')