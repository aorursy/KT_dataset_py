import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter as counter

data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')

print(data_train.columns)
print(data_test.columns)
data_train.SalePrice.describe()
data_test['SalePrice'] = 0
data = pd.concat([data_train, data_test], axis = 0).reset_index()
print('Shape of this dataset is',data.shape,'\n')
print('Following are the first five rows of the dataset: \n \n',data.head(),'\n \n')
print('The columns are:',data.columns)
data.isna().sum()
null_cols = data.columns[data.isna().any()]
print(null_cols)
data[null_cols].isna().sum()
fixable_null_cols = ['Alley','BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
       'FireplaceQu', 'GarageType','GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
       'MiscFeature']
data[fixable_null_cols].dtypes
data[fixable_null_cols] = data[fixable_null_cols].fillna('none')
new_null_cols = data.columns[data.isna().any()]
print(new_null_cols)
data[new_null_cols].isna().sum()
data[new_null_cols].dtypes
counter(data.MasVnrType)
counter(data.Electrical)
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['Electrical'] = data['Electrical'].fillna('SBrkr')
100 * data[new_null_cols].isna().sum()['LotFrontage'] / len(data) 
del data['LotFrontage']
data['MasVnrArea'] = data['MasVnrArea'].fillna((data['MasVnrArea'].median()))
data['GarageYrBlt'] = data['GarageYrBlt'].fillna((data['GarageYrBlt'].median()))
data.isna().any().any()
train = data[data.SalePrice > 0].reset_index(drop = True)
test = data[data.SalePrice == 0].reset_index(drop = True)

del test['SalePrice']
train.to_csv('train_no_missing.csv')
test.to_csv('test_no_missing.csv')