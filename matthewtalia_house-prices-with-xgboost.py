# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import math
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
# Load into Pandas dataframe
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

# Create copies of train and test
train_c = train.drop(['Id'],axis=1)
test_c = test.drop(['Id'],axis=1)
train_c.head()
test_c.head()
# This is recommended by Author
train_c = train_c.drop(train_c[(train_c['GrLivArea']>4000)].index)

# Remove heavily skewed values
train_c = train_c[train_c['SalePrice'] < train_c['SalePrice'].quantile(0.95)]
train_c['SalePrice'].skew()
targets = train_c.SalePrice
train_c = train_c.drop(['SalePrice'],axis=1)
# Training set = 1, Test set = 0
train_c['DataType'] = 1 
test_c['DataType'] = 0
data = pd.concat([train_c, test_c], axis=0,sort=False)
data.head(-10)
data.info()
percentage = data.isna().sum()*100/data.shape[0]
percentage.sort_values(ascending=False).head()
# Drop features with large number of missing values
data = data.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1)
data_num = data.select_dtypes(exclude=['object'])
data_cat = data.select_dtypes(include=['object'])

# MSSubClass should be categorical
data_num = data_num.drop(['MSSubClass'],axis=1)
data_cat['MSSubClass'] = data['MSSubClass']
data_num.isnull().sum().sort_values(ascending=False)
data_num['TotalPorch'] = data_num['OpenPorchSF'] + data_num['EnclosedPorch'] + data_num['3SsnPorch'] + data_num['ScreenPorch']
data_num['TotalFlrSF'] = data_num['1stFlrSF'] + data_num['2ndFlrSF']
data_num['TotalBsmtBath'] = data_num['BsmtFullBath'] + 0.5*data_num['BsmtHalfBath']
data_num['TotalBath'] = data_num['FullBath'] + 0.5*data_num['HalfBath']

data_num['YrSold'].loc[data_num['YrSold'] < data_num['YearBuilt']] = 2009
data_num['YearRemodAdd'].loc[data_num['YearBuilt'] > data_num['YearRemodAdd']] = 2002
# Replace these columns with median values
data_num['LotFrontage'].fillna(data_num['LotFrontage'].median(), inplace=True)
data_num['GarageYrBlt'].fillna(data_num['GarageYrBlt'].median(), inplace=True)

# Replace rest with just zeros
data_num.fillna(0, inplace=True)
data_num.skew()[abs(data_num.skew()) > 1].sort_values(ascending=False)
skewed = data_num.skew()[abs(data_num.skew()) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewed.index
lam = 0.15
for feature in skewed_features:
    data_num[feature] = boxcox1p(data_num[feature], lam)
data_cat.isnull().sum().sort_values(ascending=False)
# Columns to fill with None
col = ['FireplaceQu','GarageQual','GarageCond','GarageFinish','GarageType','BsmtCond','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinType2']
data_cat[col] = data_cat[col].fillna('None')

# Columns to fill with most frequent
col = ['MasVnrType','MSZoning','Utilities','Functional','Electrical','SaleType','Exterior2nd','Exterior1st','KitchenQual']
data_cat[col] = data_cat[col].fillna(data_cat[col].mode().iloc[0])
mapper = {
    'Reg':4, 
    'IR1':3,
    'IR2':2,
    'IR3':1,
          
    'Lvl':4,
    'Bnk':3,
    'HLS':2,
    'Low':1,
    
    'AllPub':4,
    'NoSewr':3,
    'NoSeWa':2,
    'ELO':1,
    
    'Gtl':3,
    'Mod':2,
    'Sev':1,
    
    'Ex':5,
    'Gd':4,
    'TA':3,
    'Fa':2,
    'Po':1,
    'None':0,
    
    'GLQ':6,
    'ALQ':5,
    'BLQ':4,
    'Rec':3,
    'LwQ':2,
    'Unf':1,
    
    'SBrkr':5,
    'FuseA':4,
    'FuseF':3,
    'FuseP':2,
    'Mix':1,
    
    'Typ':8,
    'Min1':7,
    'Min2':6,
    'Mod':5,
    'Maj1':4,
    'Maj2':3,
    'Sev':2,
    'Sal':1,
    
    'Fin':3,
    'RFn':2,
    'Unf':1   
}
data_cat = data_cat.replace(mapper)

# Deal with PavedDrive seperately
pave = {"N" : 0, "P" : 1, "Y" : 2}
data_cat['PavedDrive'] = data_cat['PavedDrive'].replace(pave)
# Get dummies for rest of categorical data
data_cat_obj = data_cat.select_dtypes(include=['object'])
data_cat = pd.get_dummies(data_cat, columns=data_cat_obj.columns)
data_cat = pd.get_dummies(data_cat, columns=['MSSubClass'])
data_cat.head()
data_all = pd.concat([data_num, data_cat], axis=1,sort=False)
data_train = data_all[data_all['DataType'] == 1]
data_test = data_all[data_all['DataType'] == 0]
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

X_train,X_test,y_train,y_test = train_test_split(data_train,targets,test_size=0.25,random_state=42)
import xgboost as xgb
xgbr = xgb.XGBRegressor(n_estimators=1000,
                        learning_rate=0.01,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        gamma=1,
                        random_state=42,
                        verbosity=0)
xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)
msle = math.sqrt(mean_squared_log_error(y_test, y_pred))
print("Mean Squared Log Error : ", msle)
submission = pd.DataFrame({"Id": test["Id"], "SalePrice": xgbr.predict(data_test)})
submission.to_csv('submission.csv', index=False)
submission