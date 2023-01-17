# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import glob
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

from scipy.stats import norm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

sns.distplot(train.SalePrice, fit=norm, ax=ax1)
sns.distplot(np.log1p(train.SalePrice), fit=norm, ax=ax2)
sns.stripplot(data=np.log1p(train.SalePrice), jitter=True)
train = train.iloc[:,1:]
label = train.iloc[:,-1]
test = test.iloc[:,1:]
data = pd.concat([train,test],axis=0).drop('SalePrice',axis=1)
data['MSZoning'].mode()
data['MSZoning'] = data['MSZoning'].fillna('RL')
data['LotFrontage'].mode()
data['LotFrontage'] = data['LotFrontage'].fillna(60)
data['Alley'] = data['Alley'].fillna('None')
data['Utilities'].mode()
data['Utilities'] = data['Utilities'].fillna('AllPub')
data['Exterior1st'].mode()
data['Exterior1st'] = data['Exterior1st'].fillna('VinylSd')
data['Exterior2nd'].mode()
data['Exterior2nd'] = data['Exterior2nd'].fillna('VinylSd')
data['MasVnrType'] = data['MasVnrType'].fillna('None')
data['MasVnrArea'].median()
data['MasVnrArea'] = data['MasVnrArea'].fillna(0)
data['BsmtQual'] = data['BsmtQual'].fillna('None')
data['BsmtCond'] = data['BsmtCond'].fillna('None')
data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
data[data['BsmtFinSF1'].isna()]['BsmtFinType1']
data[all_data['BsmtFinType1']=='None']['BsmtFinSF1'].unique()
data['BsmtFinSF1'] = data['BsmtFinSF1'].fillna(0)
data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')
data[data['BsmtFinSF2'].isna()]['BsmtFinType2']
data[data['BsmtFinType2']=='None']['BsmtFinSF2'].unique()
data[data['BsmtFinType2']=='None']['BsmtFinSF2'].mode()
data['BsmtFinSF2'] = data['BsmtFinSF2'].fillna(0)
data[data['BsmtUnfSF']!=0]['BsmtUnfSF'].median()
data['BsmtUnfSF'] = data['BsmtUnfSF'].fillna(520)
data[data['TotalBsmtSF'].isna()]['BsmtUnfSF']
data['TotalBsmtSF'] = data['TotalBsmtSF'].fillna(520)
data[data['BsmtFullBath'].isna()]['BsmtFinSF1']
data['BsmtFullBath'] = data['BsmtFullBath'].fillna(0)
data[data['BsmtHalfBath'].isna()]['BsmtFinSF1']
data['BsmtHalfBath'] = data['BsmtHalfBath'].fillna(0)
data[data['KitchenQual'].isna()]['KitchenAbvGr']
data[data['KitchenAbvGr']==1]['KitchenQual'].mode()
data['KitchenQual'] = data['KitchenQual'].fillna('TA')
data['Functional'].mode()
data['Functional'] = data['Functional'].fillna('Typ')
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
data['GarageType'] = data['GarageType'].fillna('None')

data['GarageYrBlt'].mode()
data[data['GarageYrBlt']==2005]['GarageYrBlt'].count()
data['GarageYrBlt'] = data['GarageYrBlt'].fillna(2005)
data['Electrical'].mode()
data['Electrical'] = data['Electrical'].fillna('SBrkr')
data['GarageFinish'] = data['GarageFinish'].fillna('None')
data['GarageQual'] = data['GarageQual'].fillna('None')
data['GarageCond'] = data['GarageCond'].fillna('None')
data['PoolQC'] = data['PoolQC'].fillna('None')
data['Fence'] = data['Fence'].fillna('None')
data['MiscFeature'] = data['MiscFeature'].fillna('None')
data[data['SaleType'].isna()]
data[(data['MSSubClass']==20) & (data['MSZoning']=='RL') &
         (data['YearBuilt']<1960) & (data['YrSold']==2007)]['SaleType'].mode()
data['SaleType'] = data['SaleType'].fillna('WD')
data[data['GarageCars'].isna()]['GarageFinish']
data['GarageCars'] = data['GarageCars'].fillna(0)
data[data['GarageArea'].isna()]['GarageFinish']
data['GarageArea'] = data['GarageArea'].fillna(0)
data.isna().sum()
train = all_data.iloc[:1460,:]
test = all_data.iloc[1460:,:]
train_str = train.select_dtypes(include=['object'])
train_int = train.select_dtypes(exclude=['object'])
test_str = test.select_dtypes(include=['object'])
test_int = test.select_dtypes(exclude=['object'])
from sklearn import preprocessing
encs = preprocessing.LabelEncoder()
for i in range(train_str.shape[1]):
    encs.fit(train_str.iloc[:,i])
    train_str.iloc[:,i] = encs.transform(train_str.iloc[:,i])
train = pd.concat([train_int,train_str],axis=1)
for i in range(test_str.shape[1]):
    encs.fit(test_str.iloc[:,i])
    test_str.iloc[:,i] = encs.transform(test_str.iloc[:,i])
test = pd.concat([test_int,test_str],axis=1)
from sklearn import svm
from xgboost import XGBRegressor,plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,log_loss
import matplotlib.pyplot as plt
good_feats = np.abs(np.array(sgd_l1.coef_)) > 1e+12


train_l1_select = train.loc[:, good_feats]
test_l1_select = test.loc[:, good_feats]


model = XGBRegressor(max_depth=5,n_estimators=1000,learning_rate=1e-2)
model.fit(train,label)
plot_importance(model,max_num_features=25)
plt.show()
good_feats_xgb = np.abs(np.array(model.feature_importances_)) > 1e-2
print("Features reduced from %10d to %10d" % (train.shape[1], int(good_feats_xgb.sum())))
train_xgb_select = train.loc[:, good_feats_xgb]
test_xgb_select = test.loc[:, good_feats_xgb]
x_train,x_val,y_train,y_val = train_test_split(train_xgb_select,label,test_size = 0.3,random_state = 3)
model = XGBRegressor(max_depth=5,learning_rate=1e-2,n_estimators=1000)
model.fit(x_train,y_train,eval_metric='rmse',eval_set=[(x_val,y_val)],early_stopping_rounds=200,verbose=False)
pred = model.predict(test_xgb_select)

sample_submission.SalePrice = pred
sample_submission.to_csv('submission2.csv',index=False)