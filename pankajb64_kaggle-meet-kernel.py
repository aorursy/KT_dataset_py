# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #data science plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.

#Ref - https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
dataFrame_train = pd.read_csv('../input/train.csv')
dataFrame_test = pd.read_csv('../input/test.csv')
dataFrame = pd.concat([dataFrame_train, dataFrame_test])
dataFrame.head(5)
y = dataFrame_train['SalePrice']
dataFrame = dataFrame.drop(['SalePrice', 'Utilities', 'Id'], axis=1)
dataFrame.info()
dataFrame.columns[dataFrame.isna().any()].tolist()
dataFrame.Alley.fillna('NA', inplace=True)
dataFrame.BsmtCond.fillna('NA', inplace=True)
dataFrame.BsmtExposure.fillna('NA', inplace=True)
dataFrame.BsmtFinSF1.fillna(0, inplace=True)
dataFrame.BsmtFinSF2.fillna(0, inplace=True)
dataFrame.BsmtFinType1.fillna('NA', inplace=True)
dataFrame.BsmtFinType2.fillna('NA', inplace=True)
dataFrame.BsmtFullBath.fillna(0, inplace=True)
dataFrame.BsmtHalfBath.fillna(0, inplace=True)
dataFrame.BsmtQual.fillna("NA", inplace=True)
dataFrame.BsmtUnfSF.fillna(0, inplace=True)
dataFrame.Electrical.fillna('SBrkr', inplace=True)
dataFrame.Exterior1st.fillna('VinylSd', inplace=True)
dataFrame.Exterior2nd.fillna('VinylSd', inplace=True)
dataFrame.Fence.fillna('NA', inplace=True)
dataFrame.FireplaceQu.fillna('NA', inplace=True)
dataFrame.Functional.fillna('Typ', inplace=True)
dataFrame.GarageArea.fillna(0, inplace=True)
dataFrame.GarageCond.fillna('NA', inplace=True)
dataFrame.GarageCars.fillna(0, inplace=True)
dataFrame.GarageYrBlt.fillna(0, inplace=True)
dataFrame.GarageQual.fillna("NA", inplace=True)
dataFrame.GarageFinish.fillna("NA", inplace=True)
dataFrame.GarageType.fillna("NA", inplace=True)
dataFrame.KitchenQual.fillna("TA", inplace=True)
dataFrame.LotFrontage.fillna(dataFrame.LotFrontage.mean(), inplace=True)
dataFrame.MSZoning.fillna("RL", inplace=True)
dataFrame.MasVnrType.fillna("None", inplace=True)
dataFrame.MasVnrArea.fillna(0, inplace=True)
dataFrame.MiscFeature.fillna('NA', inplace=True)
dataFrame.PoolQC.fillna('NA', inplace=True)
dataFrame.SaleType.fillna("WD", inplace=True)
dataFrame.TotalBsmtSF.fillna(0, inplace=True)
#dataFrame.Utilities.fillna('AllPub', inplace=True)
#dataFrame['houseAge'] = dataFrame['YrSold'] -( dataFrame['YearRemodAdd'] + dataFrame['YearBuilt'])/2.0
#Columns which have integer values but are actually categorical
int_to_cat_columns = ['MSSubClass']
cat_columns = dataFrame.select_dtypes(include='object').columns.values.tolist()
dataFrame = pd.get_dummies(dataFrame, columns=cat_columns+int_to_cat_columns)
dataFrame.columns.tolist()

#Rebuild train and test data frames
dataFrame_trainval = dataFrame.iloc[:y.shape[0], :]
X_trainval = dataFrame_trainval.values
dataFrame_trainval['SalePrice'] = y
dataFrame_test = dataFrame.iloc[y.shape[0]:, :]
X_test = dataFrame_test.values
splitter = ShuffleSplit(n_splits=1, test_size=0.2)
splits = list(splitter.split(X_trainval, y))[0]
train_ind, test_ind = splits
X_train = X_trainval[train_ind]
X_val  = X_trainval[test_ind]

y_train = y[train_ind]
y_val  = y[test_ind]

xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.12,subsample=0.5)
y_log_train = np.log(y_train)
y_log_val = np.log(y_val)
xgb.fit(X_train, y_log_train, eval_set=[(X_val, y_log_val)], verbose=True)
y_pred = np.exp(xgb.predict(X_test))
res = pd.DataFrame()
res['Id'] = dataFrame_test['Id']
res['SalePrice'] = y_pred
res.to_csv('rf.csv', index=False)
#https://towardsdatascience.com/why-automated-feature-engineering-will-change-the-way-you-do-machine-learning-5c15bf188b96
