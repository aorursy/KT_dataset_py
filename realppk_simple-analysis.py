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
# Input the test and train data
data_raw = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
# Print the first Five records of Train Data
data_raw.head()
# Print the first Five records of Test Data
data_raw.head()
# Printing Shape of Train and Test Data
print(data_raw.shape)
print(data_test.shape)
from scipy import stats
target = data_raw.SalePrice.values
xtarget, lambda_prophet = stats.boxcox(data_raw['SalePrice'] + 1)
data_raw.drop(['SalePrice'],axis=1, inplace=True)
len_train=xtarget.shape[0]
merged_df = pd.concat([data_raw, data_test])
merged_df
# Finding Columns with null values
null_data = merged_df.isna().sum()
null_data[null_data>0]
# Removing null values in the DataFrame
merged_df.loc[merged_df['LotFrontage'].isnull(),['LotFrontage']]=0.0
merged_df.loc[merged_df['Alley'].isnull(),['Alley']]='None'
merged_df.loc[merged_df['MasVnrType'].isnull(),['MasVnrType']]='None'
merged_df.loc[merged_df['MasVnrArea'].isnull(),['MasVnrArea']]=0.0e1
merged_df.loc[merged_df['BsmtQual'].isnull(),['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']]='None'
merged_df.loc[merged_df['BsmtFinType1'].isnull(),['BsmtFinType1']]='None'
merged_df.loc[merged_df['BsmtFinType2'].isnull(),['BsmtFinType2']]='None'
merged_df.loc[merged_df['BsmtExposure'].isnull(),['BsmtExposure']]='None'
merged_df.loc[merged_df['BsmtCond'].isnull(),['BsmtCond']]='None'
merged_df.loc[merged_df['BsmtExposure']=='No',['BsmtExposure']]='None'
merged_df.loc[merged_df['BsmtFinSF1'].isnull(),['BsmtFinSF1']]=0.0
merged_df.loc[merged_df['BsmtFinSF2'].isnull(),['BsmtFinSF2']]=0.0
merged_df.loc[merged_df['TotalBsmtSF'].isnull(),['TotalBsmtSF']]=0.0
merged_df.loc[merged_df['BsmtUnfSF'].isnull(),['BsmtUnfSF']]=0.0
merged_df.loc[merged_df['BsmtFullBath'].isnull(),['BsmtFullBath']]=0.0
merged_df.loc[merged_df['BsmtHalfBath'].isnull(),['BsmtHalfBath']]=0.0
merged_df.loc[merged_df['GarageType'].isnull(),['GarageType','GarageFinish','GarageYrBlt','GarageQual','GarageCond']]='None'
merged_df.loc[merged_df['GarageFinish'].isnull(),['GarageFinish']]='None'
merged_df.loc[merged_df['GarageCars'].isnull(),['GarageCars']]=0.
merged_df.loc[merged_df['GarageArea'].isnull(),['GarageArea']]=0.0
merged_df.loc[merged_df['GarageQual'].isnull(),['GarageQual']]='None'
merged_df.loc[merged_df['GarageCond'].isnull(),['GarageCond']]='None'
merged_df.loc[merged_df['GarageYrBlt'].isnull(),['GarageYrBlt']]='None'
merged_df.loc[merged_df['MSZoning'].isnull(),['MSZoning']]='RL'
merged_df.loc[merged_df['Utilities'].isnull(),['Utilities']]='None'
merged_df.loc[merged_df['FireplaceQu'].isnull(),['FireplaceQu']]='None'
merged_df.loc[merged_df['Exterior1st'].isnull(),['Exterior1st']]='VinylSd'
merged_df.loc[merged_df['Exterior2nd'].isnull(),['Exterior2nd']]='VinylSd'
merged_df.loc[merged_df['PoolQC'].isnull(),['PoolQC']]='None'
merged_df.loc[merged_df['Fence'].isnull(),['Fence']]='None'
merged_df.loc[merged_df['MiscFeature'].isnull(),['MiscFeature']]='None'
merged_df.loc[merged_df['Electrical'].isnull(),['Electrical']]='SBrkr'
merged_df.loc[merged_df['KitchenQual'].isnull(),['KitchenQual']]='TA'
merged_df.loc[merged_df['Functional'].isnull(),['Functional']]='Typ'
merged_df.loc[merged_df['SaleType'].isnull(),['SaleType']]='WD'
# Finding Columns with null values
null_data = merged_df.isna().sum()
null_data[null_data>0]
merged_df.drop(['Id'],axis=1, inplace=True)
# Identifying Columns with Categorical Data
Columns = merged_df.select_dtypes(include='O').columns.values
print("Before OHE", merged_df.shape)
merged_df = pd.get_dummies(merged_df, columns= Columns)
print("After OHE", merged_df.shape)
X = merged_df[:len_train].values
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X, xtarget)
print(model.feature_importances_)
import matplotlib.pyplot as plt
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()
scores = model.feature_importances_
df=pd.DataFrame(data=scores)
list_col = df.ix[df[0]>=0.001,:].index.values
merged_df = merged_df.iloc[:,list_col]
X = merged_df[:len_train].values
test_x = merged_df[len_train:].values
model = XGBRegressor()
model.fit(X, xtarget)
y_test = model.predict(test_x)
y_test
def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
y_test = inverse_boxcox(y_test, lambda_prophet) - 1
y_test
final_df = pd.DataFrame(data = y_test)
final_df.head()
final_df['SalePrice'] = final_df[0]
final_df.head()
final_df['id'] = range(1461,2920)
final_df=final_df.loc[:,['id','SalePrice']]
final_df.head()
final_df.to_csv('sample.csv',index=False)
