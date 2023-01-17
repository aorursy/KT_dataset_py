# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
X_train_full = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col='Id')

X_test_full = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col='Id')

X_train_full.head()
X_train_full.isnull().sum()
sns.heatmap(X_train_full.isnull(),yticklabels=False,cbar=False)
X_train_full.info()
X_train_full.shape
X_train_full.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
X_train_full['LotFrontage']=X_train_full['LotFrontage'].fillna(X_train_full['LotFrontage'].mean())

X_train_full['BsmtCond']=X_train_full['BsmtCond'].fillna(X_train_full['BsmtCond'].mode()[0])

X_train_full['BsmtQual']=X_train_full['BsmtQual'].fillna(X_train_full['BsmtQual'].mode()[0])

X_train_full['FireplaceQu']=X_train_full['FireplaceQu'].fillna(X_train_full['FireplaceQu'].mode()[0])

X_train_full['GarageType']=X_train_full['GarageType'].fillna(X_train_full['GarageType'].mode()[0])

X_train_full['GarageFinish']=X_train_full['GarageFinish'].fillna(X_train_full['GarageFinish'].mode()[0])

X_train_full['GarageQual']=X_train_full['GarageQual'].fillna(X_train_full['GarageQual'].mode()[0])

X_train_full['GarageCond']=X_train_full['GarageCond'].fillna(X_train_full['GarageCond'].mode()[0])
X_train_full.drop(['GarageYrBlt'],axis=1,inplace=True)
sns.heatmap(X_train_full.isnull(),yticklabels=False,cbar=False)
X_train_full['BsmtExposure']=X_train_full['BsmtExposure'].fillna(X_train_full['BsmtExposure'].mode()[0])

X_train_full['BsmtFinType2'].isnull().sum()
X_train_full['BsmtFinType2']=X_train_full['BsmtFinType2'].fillna(X_train_full['BsmtFinType2'].mode()[0])

X_train_full.info()
X_train_full['BsmtExposure']=X_train_full['BsmtExposure'].fillna(X_train_full['BsmtExposure'].mode()[0])

X_train_full['BsmtFinType1']=X_train_full['BsmtFinType1'].fillna(X_train_full['BsmtFinType1'].mode()[0])

X_train_full['BsmtFinType2']=X_train_full['BsmtFinType2'].fillna(X_train_full['BsmtFinType2'].mode()[0])

X_train_full['MasVnrType']=X_train_full['MasVnrType'].fillna(X_train_full['MasVnrType'].mode()[0])

X_train_full['MasVnrArea']=X_train_full['MasVnrArea'].fillna(X_train_full['MasVnrArea'].mode()[0])
X_train_full.dropna(inplace=True)
X_train_full.head()
X_train_full.shape
X_test_full.info()
X_test_full.drop(['Alley','PoolQC','Fence','MiscFeature','GarageYrBlt'],axis=1,inplace=True)
sns.heatmap(X_test_full.isnull(),yticklabels=False,cbar=False)
X_test_full.isnull().any()
X_test_full['LotFrontage']=X_test_full['LotFrontage'].fillna(X_test_full['LotFrontage'].mean())

cols=[col for col in X_test_full.columns if X_test_full[col].isnull().any()]
for col in cols:

    X_test_full[col]=X_test_full[col].fillna(X_test_full[col].mode()[0])
sns.heatmap(X_test_full.isnull(),yticklabels=False,cbar=False)
X_test_full.shape
# Remove rows with missing target, separate target from predictors

X_train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_train_full.SalePrice              

X_train_full.drop(['SalePrice'], axis=1, inplace=True)
from sklearn.preprocessing import OneHotEncoder



# Get list of categorical variables

s = (X_train_full.dtypes == 'object')

object_cols = list(s[s].index)



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_full[object_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test_full[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train_full.index

OH_cols_test.index = X_test_full.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train_full.drop(object_cols, axis=1)

num_X_test = X_test_full.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
from xgboost import XGBRegressor

model = XGBRegressor(learning_rate=0.01, n_estimators=3460,

                     max_depth=3, min_child_weight=0,

                     gamma=0, subsample=0.7,

                     colsample_bytree=0.7,

                     objective='reg:squarederror', nthread=-1,

                     scale_pos_weight=1, seed=27,

                     reg_alpha=0.00006)

model.fit(OH_X_train, y)

preds_test = model.predict(OH_X_test)



output = pd.DataFrame({'Id': OH_X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)