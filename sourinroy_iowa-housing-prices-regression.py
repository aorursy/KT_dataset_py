# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv',index_col = "Id")

test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv',index_col = "Id")

#rel = train_data.corr()

#rel['SalePrice'].sort_values()
train_data['HouseAge'] = train_data['YrSold'] - train_data['YearBuilt']

train_data['temp'] = (train_data['YearRemodAdd']-train_data['YearBuilt'])

train_data["Remod"] = train_data['temp'].replace( (train_data['temp'].where(train_data['temp'] > 0)) , 1 )

train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean(), inplace=True)

train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mean(), inplace = True)

train_data['Street'].replace({'Pave': 1, 'Grvl': 0}, inplace=True)

train_data['Alley'].replace({'Pave': 1, 'Grvl': 0}, inplace=True)

train_data['Alley'].fillna(-1, inplace = True)

train_data['LotShape'].replace({'Reg': 3, 'IR1': 2, 'IR2' : 1, 'IR3' : 0}, inplace=True)

train_data['Utilities'].replace({'AllPub' : 4, 'NoSeWa' : 2},inplace=True)

train_data['LandSlope'].replace({'Gtl': 0, 'Mod': -1, 'Sev' : -2}, inplace=True)

train_data['LotConfig'].replace({'FR3': 3, 'FR2': 2, 'CulDSac' : 1, 'Corner' : 0, "Inside" : -1}, inplace=True)

train_data['ExterQual'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace=True)

train_data['PoolQC'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2, 'Fa' : 1}, inplace=True)

train_data['PoolQC'].fillna(0, inplace = True)

train_data['Fence'].replace({'GdPrv': 4, 'MnPrv': 3, 'GdWo' : 2, 'MnWw' : 1}, inplace=True)

train_data['Fence'].fillna(0, inplace = True)

train_data['ExterCond'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace=True)

train_data['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['BsmtQual'].fillna(0,inplace = True)

train_data['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['BsmtCond'].fillna(0,inplace = True)

train_data['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn' : 2,'No' : 1}, inplace=True)

train_data['BsmtExposure'].fillna(0,inplace = True)

train_data['HeatingQC'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['CentralAir'].replace({'Y' : 1 , 'N' : 0}, inplace=True)

train_data['KitchenQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['Functional'].replace({'Typ': 0, 'Min1': -0.25, 'Min2' : -0.5,'Mod' : -1, 'Maj1' : -2, 'Maj2' : -3,'Sev' : -4, 'Sal' : -5}, inplace=True)

train_data['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['FireplaceQu'].fillna(0,inplace = True)

train_data['GarageType'].replace({'2Types' : 6 , 'Attchd' : 5 , 'Basment' : 4 , 'BuiltIn' : 3 , 'CarPort' : 2 , 'Detchd' : 1}, inplace=True)

train_data['GarageType'].fillna(0,inplace = True)

train_data['GarageFinish'].replace({ 'Fin' : 3, 'RFn' : 2, 'Unf' : 1}, inplace = True)

train_data['GarageFinish'].fillna(0, inplace = True)

train_data['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['GarageQual'].fillna(0,inplace = True)

train_data['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

train_data['GarageCond'].fillna(0,inplace = True)

train_data['PavedDrive'].replace({ 'Y' : 2, 'P' : 1, 'N' : 0}, inplace = True)



train_data['ExterQual'] = train_data['ExterQual'].astype(str).astype(int)

train_data['KitchenQual'] = train_data['KitchenQual'].astype(str).astype(int)

train_data['Functional'] = train_data['Functional'].astype(str).astype(float)

test_data['GarageArea'].fillna(test_data['GarageArea'].mean(),inplace = True)

test_data['GarageCars'].fillna(0,inplace = True)

test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].mean(),inplace = True)

test_data['HouseAge'] = test_data['YrSold'] - test_data['YearBuilt']

test_data['temp'] = (test_data['YearRemodAdd']-test_data['YearBuilt'])

test_data["Remod"] = test_data['temp'].replace( (test_data['temp'].where(test_data['temp'] > 0)) , 1 )

test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean(), inplace=True)

test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean(), inplace = True)

test_data['BsmtFullBath'].fillna(0, inplace = True)

test_data['BsmtHalfBath'].fillna(0, inplace = True)

test_data['Street'].replace({'Pave': 1, 'Grvl': 0}, inplace=True)

test_data['Alley'].replace({'Pave': 1, 'Grvl': 0}, inplace=True)

test_data['Alley'].fillna(-1, inplace = True)

test_data['LotShape'].replace({'Reg': 3, 'IR1': 2, 'IR2' : 1, 'IR3' : 0}, inplace=True)

test_data['Utilities'].fillna(4, inplace=True)

test_data['Utilities'].replace({'AllPub' : 4},inplace=True)

test_data['LandSlope'].replace({'Gtl': 0, 'Mod': -1, 'Sev' : -2}, inplace=True)

test_data['LotConfig'].replace({'FR3': 3, 'FR2': 2, 'CulDSac' : 1, 'Corner' : 0, "Inside" : -1}, inplace=True)

test_data['ExterQual'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2,'Fa' : 1, 'Po' : 0}, inplace=True)

test_data['PoolQC'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2,'Fa' : 1}, inplace=True)

test_data['PoolQC'].fillna(0, inplace = True)

test_data['Fence'].replace({'GdPrv': 4, 'MnPrv': 3, 'GdWo' : 2, 'MnWw' : 1}, inplace=True)

test_data['Fence'].fillna(0, inplace = True)

test_data['ExterCond'].replace({'Ex': 4, 'Gd': 3, 'TA' : 2, 'Fa' : 1, 'Po' : 0}, inplace=True)

test_data['BsmtQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['BsmtQual'].fillna(0,inplace = True)

test_data['BsmtCond'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['BsmtCond'].fillna(0,inplace = True)

test_data['BsmtExposure'].replace({'Gd': 4, 'Av': 3, 'Mn' : 2,'No' : 1}, inplace=True)

test_data['BsmtExposure'].fillna(0,inplace = True)

test_data['HeatingQC'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['CentralAir'].replace({'Y' : 1 , 'N' : 0}, inplace=True)

test_data['KitchenQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['KitchenQual'].fillna(3, inplace =True)

test_data['Functional'].replace({'Typ': 0, 'Min1': -0.25, 'Min2' : -0.5,'Mod' : -1, 'Maj1' : -2, 'Maj2' : -3,'Sev' : -4, 'Sal' : -5}, inplace=True)

test_data['Functional'].fillna(0,inplace = True)

test_data['FireplaceQu'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['FireplaceQu'].fillna(0,inplace = True)

test_data['GarageType'].replace({'2Types' : 6 , 'Attchd' : 5 , 'Basment' : 4 , 'BuiltIn' : 3 , 'CarPort' : 2 , 'Detchd' : 1}, inplace=True)

test_data['GarageType'].fillna(0,inplace = True)

test_data['GarageFinish'].replace({ 'Fin' : 3, 'RFn' : 2, 'Unf' : 1}, inplace = True)

test_data['GarageFinish'].fillna(0, inplace = True)

test_data['GarageQual'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['GarageQual'].fillna(0,inplace = True)

test_data['GarageCond'].replace({'Ex': 5, 'Gd': 4, 'TA' : 3,'Fa' : 2, 'Po' : 1}, inplace=True)

test_data['GarageCond'].fillna(0,inplace = True)

test_data['PavedDrive'].replace({ 'Y' : 2, 'P' : 1, 'N' : 0}, inplace = True)



test_data['ExterQual'] = test_data['ExterQual'].astype(str).astype(int)

test_data['KitchenQual'] = test_data['KitchenQual'].astype(str).astype(int)

test_data['Functional'] = test_data['Functional'].astype(str).astype(float)



features = ['1stFlrSF','MasVnrArea','OverallCond','OverallQual','LotFrontage','LotArea','TotalBsmtSF','GrLivArea',

            'TotRmsAbvGrd','Fireplaces','GarageArea','GarageCars','HouseAge','Remod','2ndFlrSF','MiscVal',

            'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','WoodDeckSF',

            'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea', 'Street' ,'Alley', 'LotShape', 

            'Utilities', 'LandSlope', 'LotConfig', 'PoolQC', 'Fence', 'ExterCond' , 'BsmtQual', 'BsmtCond',

            'BsmtExposure', 'HeatingQC' , 'CentralAir',  'FireplaceQu',  'GarageType', 'ExterQual','KitchenQual',

            'GarageFinish',  'GarageQual', 'GarageCond', 'PavedDrive', 'Functional']

test_data[features].isnull().sum().sum()
X = train_data[features]

y = train_data.SalePrice



X_test = test_data[features]



#X_test['ExterQual'].value_counts()

#X_test['KitchenQual'].value_counts()

#X_test['Functional'].value_counts()



X_test.isnull().sum().sum()



#from sklearn.model_selection import train_test_split

#X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=1)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

#naive_model = XGBRegressor()

imp_model = XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth = 3,n_jobs=8)

#naive_model.fit(X_train,y_train)

#naive_pred = naive_model.predict(X_valid)

#naive_mae = mean_absolute_error(y_valid,naive_pred)

#naive_mae
#from sklearn.ensemble import RandomForestRegressor

#Rf_model = RandomForestRegressor(n_estimators=100, criterion = "mae", random_state = 1)

#from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(imp_model,X,y,cv=5,scoring='neg_mean_absolute_error')

scores.mean()
from sklearn.model_selection import cross_val_score

#scores = -1 * cross_val_score(imp_model,X,y,cv=5,scoring='neg_mean_absolute_error')

#scores.mean()
#from sklearn.model_selection import cross_val_score

#for nest in [100,200,300,400,500,600,700,800,900,1200]:

    #for max_dpth in [3,4,5,6,7,8,9,10,11,12]:

       # imp_model = XGBRegressor(n_estimators=nest,learning_rate=0.05,max_depth = 8,n_jobs=8)

       # scores = -1 * cross_val_score(imp_model,X,y,cv=5,scoring='neg_mean_absolute_error')

      #  print(nest," ",max_dpth," ",scores.mean())

    
#Rf_model.fit(X,y)

#fin_pred = Rf_model.predict(X_test)
imp_model.fit(X,y)

fin_pred = imp_model.predict(X_test)
submit = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': fin_pred})

submit.to_csv('Iowa_submission.csv', index=False)