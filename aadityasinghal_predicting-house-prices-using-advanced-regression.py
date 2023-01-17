import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.metrics import r2_score
dataset_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

dataset_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dataset_train.head()
dataset_train.shape
dataset_test.head()
dataset_test.shape
dataset = pd.concat([dataset_train, dataset_test])
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
null_value_train = dict(dataset.isnull().sum())

for i,j in null_value_train.items():

    print(i,"==>",j)
dataset.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], axis=1, inplace=True)
dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode())

dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean())

dataset['Utilities'] = dataset['Utilities'].fillna(dataset['Utilities'].mode())

dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode())

dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode())

dataset['MasVnrType'] = dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode())

dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean())

dataset['BsmtQual'] = dataset['BsmtQual'].fillna(dataset['BsmtQual'].mode())

dataset['BsmtCond'] = dataset['BsmtCond'].fillna(dataset['BsmtCond'].mode())

dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(dataset['BsmtExposure'].mode())

dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(dataset['BsmtFinType1'].mode())

dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean())

dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(dataset['BsmtFinType2'].mode())

dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean())

dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean())

dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean())

dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode())

dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].median())

dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].median())

dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode())

dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode())        

dataset['GarageType'] = dataset['GarageType'].fillna(dataset['GarageType'].mode())

dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(dataset['GarageYrBlt'].median())

dataset['GarageFinish'] = dataset['GarageFinish'].fillna(dataset['GarageFinish'].mode())

dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].median())

dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean())

dataset['GarageQual'] = dataset['GarageQual'].fillna(dataset['GarageQual'].mode())        

dataset['GarageCond'] = dataset['GarageCond'].fillna(dataset['GarageCond'].mode())

dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(dataset['FireplaceQu'].mode())

dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode())
dataset = pd.get_dummies(dataset, drop_first=True)
dataset_train_1 = dataset.iloc[:1460, :]

dataset_test_1 = dataset.iloc[1460:, :]
y_train = dataset_train_1['SalePrice'].values

dataset_train_1 = dataset_train_1.drop('SalePrice', axis=1)

dataset_test_1 = dataset_test_1.drop('SalePrice', axis=1)
X_train = dataset_train_1.iloc[:, :].values

X_test = dataset_test_1.iloc[:, :].values
from xgboost import XGBRegressor

regressor = XGBRegressor()

regressor.fit(X_train, y_train)
y_pred_train = regressor.predict(X_train)

print(r2_score(y_train,y_pred_train))
y_pred_test = regressor.predict(X_test)
output = pd.DataFrame({'Id': dataset_test.Id, 'SalePrice': y_pred_test})

output.to_csv('my_submission_house_prediction_3.csv', index=False)

print("Your submission was successfully saved!")