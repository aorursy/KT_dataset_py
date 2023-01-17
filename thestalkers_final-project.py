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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df.head()
df['MSZoning'].value_counts()

X = df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id','Condition2','Exterior2nd','YearBuilt','Street','Utilities','SalePrice','GarageCond'],axis = 1)
X_test = df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature','Id','Condition2','Exterior2nd','YearBuilt','Street','Utilities','GarageCond'],axis = 1)

#Filling Missing Values(X_train)
X['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace = True)
df['GarageType'].value_counts()
X['GarageType'].fillna('Attchd',inplace = True)
X['GarageYrBlt'].fillna(method = 'ffill',inplace = True)
df['GarageFinish'].value_counts()
X['GarageFinish'].fillna(method = 'ffill',inplace = True)
df['GarageQual'].value_counts()
X['GarageQual'].fillna('TA',inplace = True)
df['MasVnrType'].value_counts()
X['MasVnrType'].fillna(method = 'ffill',inplace = True)
X['MasVnrArea'].fillna(X['MasVnrArea'].mean(),inplace = True)
df['BsmtQual'].value_counts()
X['BsmtQual'].fillna(method = 'ffill',inplace = True)
df['BsmtCond'].value_counts()
X['BsmtCond'].fillna('TA',inplace = True)
df['BsmtExposure'].value_counts()
X['BsmtExposure'].fillna('No',inplace = True)
df['BsmtFinType1'].value_counts()
X['BsmtFinType1'].fillna(method = 'ffill',inplace = True)
df['BsmtFinType2'].value_counts()
X['BsmtFinType2'].fillna('Unf',inplace = True)
X['Electrical'].value_counts()
X['Electrical'].fillna('SBrkr',inplace = True)
X.isna().sum().max()

#Filling Missing Values(X_test)
X_test['LotFrontage'].fillna(df_test['LotFrontage'].mean(),inplace = True)
df_test['GarageType'].value_counts()
X_test['GarageType'].fillna('Attchd',inplace = True)
X_test['GarageYrBlt'].fillna(method = 'ffill',inplace = True)
df_test['GarageFinish'].value_counts()
X_test['GarageFinish'].fillna(method = 'ffill',inplace = True)
df_test['GarageQual'].value_counts()
X_test['GarageQual'].fillna('TA',inplace = True)
df_test['MasVnrType'].value_counts()
X_test['MasVnrType'].fillna(method = 'ffill',inplace = True)
X_test['MasVnrArea'].fillna(X['MasVnrArea'].mean(),inplace = True)
df_test['BsmtQual'].value_counts()
X_test['BsmtQual'].fillna(method = 'ffill',inplace = True)
df_test['BsmtCond'].value_counts()
X_test['BsmtCond'].fillna('TA',inplace = True)
df_test['BsmtExposure'].value_counts()
X_test['BsmtExposure'].fillna('No',inplace = True)
df_test['BsmtFinType1'].value_counts()
X_test['BsmtFinType1'].fillna(method = 'ffill',inplace = True)
df_test['BsmtFinType2'].value_counts()
X_test['BsmtFinType2'].fillna('Unf',inplace = True)
X_test['Electrical'].value_counts()
X_test['Electrical'].fillna('SBrkr',inplace = True)
X_test['MSZoning'].value_counts()
X_test['MSZoning'].fillna('RL',inplace = True)
X_test['Exterior1st'].value_counts()
X_test['Exterior1st'].fillna('VinylSd',inplace = True)
X_test['BsmtFinSF1'].fillna(X_test['BsmtFinSF1'].mean(),inplace = True)
X_test['BsmtFinSF2'].fillna(X_test['BsmtFinSF2'].mean(),inplace = True)
X_test['BsmtFullBath'].value_counts()
X_test['BsmtFullBath'].fillna(0.0,inplace = True)
X_test['BsmtHalfBath'].value_counts()
X_test['BsmtHalfBath'].fillna(0.0,inplace = True)
X_test['KitchenQual'].value_counts()
X_test['KitchenQual'].fillna('TA',inplace = True)
X_test['Functional'].value_counts()
X_test['Functional'].fillna('Typ',inplace = True)
X_test['GarageCars'].value_counts()
X_test['GarageCars'].fillna(2.0,inplace = True)
X_test['SaleType'].value_counts()
X_test['SaleType'].fillna('WD',inplace = True)
X_test['GarageArea'].fillna(X_test['GarageArea'].mean(),inplace = True)
X_test['BsmtUnfSF'].value_counts()
X_test['BsmtUnfSF'].fillna(X_test['BsmtUnfSF'].mean(),inplace = True)
X_test['TotalBsmtSF'].fillna(X_test['TotalBsmtSF'].mean(),inplace = True)
X_test.isna().sum().max()

X_new = X.append(X_test)
X_new = pd.get_dummies(X_new)
X_new = X_new.drop(['MSZoning_FV','LotShape_IR1','LandContour_Bnk','LotConfig_Corner','LandSlope_Gtl','Neighborhood_BrDale','Condition1_Artery','BldgType_1Fam','HouseStyle_1.5Fin','RoofStyle_Flat','RoofMatl_Metal','Exterior1st_AsbShng','MasVnrType_None','ExterQual_Ex','ExterCond_Ex','Foundation_Slab','BsmtQual_Ex','BsmtCond_Gd','BsmtExposure_Av','BsmtFinType1_ALQ','BsmtFinType2_ALQ','Heating_Floor','CentralAir_N','Electrical_FuseA','KitchenQual_Ex','Functional_Maj1','GarageType_Attchd','GarageFinish_Fin','GarageQual_Ex','PavedDrive_N','SaleType_COD','SaleCondition_Abnorml'],axis = 1)
X = X_new[0:1460]
X_test = X_new[1460:]
Y = df['SalePrice']

from sklearn.cross_validation import train_test_split
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size = 0.1, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_cv)

y_pred_test = regressor.predict(X_test)
