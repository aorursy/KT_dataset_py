# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_hp = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train_hp.shape, test_hp.shape
train_hp.head()
train_hp.info()
train_hp.describe()
plt.figure(figsize = (12,8))
sns.heatmap(train_hp.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.hist(train_hp.SalePrice, color='blue')
plt.show()
train_hp.SalePrice.skew()
num_hp = train_hp.select_dtypes(include=[np.number])
num_hp.dtypes
fig= plt.subplots(figsize=(12,12)) 
sns.heatmap(num_hp.corr(),cmap=sns.diverging_palette(20, 220, n=200), linewidths=2)
train_hp.OverallQual.unique()
qual_pivot = train_hp.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.mean)
qual_pivot
qual_pivot.plot(kind = 'bar', color='grey')
sns.boxplot(x=train_hp['OverallQual'], y=train_hp.SalePrice)
plt.ylabel('Sale Price')
plt.xlabel('OverallQual')
plt.show()
plt.scatter(x=train_hp['GrLivArea'], y=train_hp.SalePrice)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
null_data = pd.DataFrame(train_hp.isnull().sum().sort_values(ascending=False))
null_data.columns = ['Null Count']
null_data.drop(null_data[null_data['Null Count'] == 0].index, inplace =True)
percent = (train_hp.isnull().sum()/train_hp.isnull().count()).sort_values(ascending=False)*100
null_data['percent'] = percent
null_data
train_hp['PoolQC'].fillna('No Pool', inplace=True)
train_hp['MiscFeature'].fillna('No Other Features', inplace=True)
train_hp['Alley'].fillna('No Alley Access', inplace =True)
train_hp['Fence'].fillna('No Fence', inplace =True)
train_hp['FireplaceQu'].fillna('No Fireplace', inplace =True)
train_hp['LotFrontage'].fillna(train_hp['LotFrontage'].mean(), inplace =True)
train_hp['GarageQual'].fillna('No Garage', inplace =True)
train_hp['GarageCond'].fillna('No Garage', inplace =True)
train_hp['GarageType'].fillna('No Garage', inplace =True)
train_hp['GarageFinish'].fillna('No Garage', inplace =True)
train_hp['BsmtFinType2'].fillna('No Basement', inplace =True)
train_hp['BsmtCond'].fillna('No Basement', inplace =True)
train_hp['BsmtExposure'].fillna('No Basement', inplace =True)
train_hp['BsmtQual'].fillna('No Basement', inplace =True)
train_hp['BsmtFinType1'].fillna('No Basement', inplace =True)
train_hp['MasVnrArea'].fillna(0, inplace =True)
train_hp['MasVnrType'].fillna('No Masonary', inplace =True)
train_hp['Electrical'].fillna('SBrkr', inplace =True)
train_hp['GarageYrBlt'].fillna(train_hp['GarageYrBlt'].mean(), inplace =True)
hpmodel=train_hp.drop(["TotRmsAbvGrd", "1stFlrSF", "GarageCars"], axis=1)
total = hpmodel.isnull().sum().sort_values(ascending=False)
percent = (hpmodel.isnull().sum()/train_hp.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(18)
nulls = pd.DataFrame(test_hp.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null']
nulls.drop(nulls[nulls['Null'] == 0].index, inplace =True)
percent = (test_hp.isnull().sum()/train_hp.isnull().count()).sort_values(ascending=False)*100
nulls['percent'] = percent

test_hp['PoolQC'].fillna('No Pool', inplace=True)
test_hp['MiscFeature'].fillna('No Other Features', inplace=True)
test_hp['Alley'].fillna('No Alley Access', inplace =True)
test_hp['Fence'].fillna('No Fence', inplace =True)
test_hp['FireplaceQu'].fillna('No Fireplace', inplace =True)
test_hp['LotFrontage'].fillna(train_hp['LotFrontage'].mean(), inplace =True)
test_hp['GarageQual'].fillna('No Garage', inplace =True)
test_hp['GarageCond'].fillna('No Garage', inplace =True)
test_hp['GarageType'].fillna('No Garage', inplace =True)
test_hp['GarageFinish'].fillna('No Garage', inplace =True)
test_hp['BsmtFinType2'].fillna('No Basement', inplace =True)
test_hp['BsmtCond'].fillna('No Basement', inplace =True)
test_hp['BsmtExposure'].fillna('No Basement', inplace =True)
test_hp['BsmtQual'].fillna('No Basement', inplace =True)
test_hp['BsmtFinType1'].fillna('No Basement', inplace =True)
test_hp['MasVnrArea'].fillna(0, inplace =True)
test_hp['MasVnrType'].fillna('No Masonary', inplace =True)
test_hp['Electrical'].fillna('SBrkr', inplace =True)
test_hp['GarageYrBlt'].fillna(train_hp['GarageYrBlt'].mean(), inplace =True)
test_hp['MSZoning'].fillna('No MSZoning', inplace=True)
test_hp['BsmtHalfBath'].fillna(train_hp['BsmtHalfBath'].mean(), inplace=True)
test_hp['Utilities'].fillna('No Utilities', inplace =True)
test_hp['Functional'].fillna('No Functional', inplace =True)
test_hp['BsmtFullBath'].fillna(train_hp['BsmtFullBath'].mean(), inplace =True)
test_hp['BsmtFinSF2'].fillna(train_hp['BsmtFinSF2'].mean(), inplace =True)
test_hp['BsmtFinSF1'].fillna(train_hp['BsmtFinSF1'].mean(), inplace =True)
test_hp['Exterior2nd'].fillna('Exterior2nd', inplace =True)
test_hp['BsmtUnfSF'].fillna(train_hp['BsmtUnfSF'].median(), inplace =True)
test_hp['TotalBsmtSF'].fillna(train_hp['TotalBsmtSF'].median(), inplace =True)
test_hp['SaleType'].fillna('No SaleType', inplace =True)
test_hp['Exterior1st'].fillna('No Exterior1st', inplace =True)
test_hp['KitchenQual'].fillna('No KitchenQual', inplace =True)
test_hp['GarageArea'].fillna(train_hp['GarageArea'].median(), inplace =True)
test_hp['GarageCars'].fillna(train_hp['GarageCars'].mean(), inplace =True)                             
hpmodel_test =test_hp.drop(["TotRmsAbvGrd", "1stFlrSF"], axis=1)
total1 = hpmodel_test.isnull().sum().sort_values(ascending=False)
missing_data_test.head(18)
cat_hp = train_hp.select_dtypes('object').columns
cat_hp
train_new = pd.get_dummies(train_hp,columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition'])
cat_test = test_hp.select_dtypes('object').columns
cat_test
test_new = pd.get_dummies(test_hp,columns=['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
       'SaleType', 'SaleCondition'])
y = train_new.SalePrice
X = train_new.drop(['SalePrice', 'Id'], axis=1) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size= 0.33)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
Y_pred=lr.predict(X_test)
coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
coeff_df
lr.score(X_test, y_test)
from sklearn import metrics
from math import sqrt
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, Y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, Y_pred)))
print('RMSE: {}'.format(sqrt(metrics.mean_squared_error(y_test, Y_pred))))
print("R2: {}".format(metrics.r2_score(y_test,Y_pred)))
my_submission = pd.DataFrame({'Id': test_new.Id})
my_submission.to_csv('firstsubmission.csv', index=False)