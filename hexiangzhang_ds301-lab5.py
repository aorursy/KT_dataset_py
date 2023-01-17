import numpy as np

import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
y = df.SalePrice
from sklearn.preprocessing import LabelEncoder



df['MSZoning'] = LabelEncoder().fit_transform(df['MSZoning'])

df['Street'] = LabelEncoder().fit_transform(df['Street'])

df['LotShape'] = LabelEncoder().fit_transform(df['LotShape'])

df['LandContour'] = LabelEncoder().fit_transform(df['LandContour'])

df['Utilities'] = LabelEncoder().fit_transform(df['Utilities'])

df['LotConfig'] = LabelEncoder().fit_transform(df['LotConfig'])

df['LandSlope'] = LabelEncoder().fit_transform(df['LandSlope'])

df['Neighborhood'] = LabelEncoder().fit_transform(df['Neighborhood'])

df['Condition1'] = LabelEncoder().fit_transform(df['Condition1'])

df['Condition2'] = LabelEncoder().fit_transform(df['Condition2'])

df['BldgType'] = LabelEncoder().fit_transform(df['BldgType'])

df['RoofMatl'] = LabelEncoder().fit_transform(df['RoofMatl'])

df['RoofStyle'] = LabelEncoder().fit_transform(df['RoofStyle'])

df['Exterior1st'] = LabelEncoder().fit_transform(df['Exterior1st'])

df['Exterior2nd'] = LabelEncoder().fit_transform(df['Exterior2nd'])

df['ExterQual'] = LabelEncoder().fit_transform(df['ExterQual'])

df['ExterCond'] = LabelEncoder().fit_transform(df['ExterCond'])

df['Foundation'] = LabelEncoder().fit_transform(df['Foundation'])

df['Heating'] = LabelEncoder().fit_transform(df['Heating'])

df['HeatingQC'] = LabelEncoder().fit_transform(df['HeatingQC'])

df['CentralAir'] = LabelEncoder().fit_transform(df['CentralAir'])

df['KitchenQual'] = LabelEncoder().fit_transform(df['KitchenQual'])

df['Functional'] = LabelEncoder().fit_transform(df['Functional'])

df['PavedDrive'] = LabelEncoder().fit_transform(df['PavedDrive'])

df['SaleType'] = LabelEncoder().fit_transform(df['SaleType'])

df['SaleCondition'] = LabelEncoder().fit_transform(df['SaleCondition'])
X=df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 

             'YearRemodAdd', 'RoofMatl', 'Street', 'LotShape', 'LandContour', 'LotConfig', 

             'RoofStyle', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 

             'BldgType', 'ExterQual', 'ExterCond', 'Foundation','MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 

             'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 

             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',  

             'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 

             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PavedDrive', 'SaleCondition', 

             'YrSold']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
from sklearn.preprocessing import LabelEncoder



df['Street'] = LabelEncoder().fit_transform(df['Street'])

df['LotShape'] = LabelEncoder().fit_transform(df['LotShape'])

df['LandContour'] = LabelEncoder().fit_transform(df['LandContour'])

df['LotConfig'] = LabelEncoder().fit_transform(df['LotConfig'])

df['LandSlope'] = LabelEncoder().fit_transform(df['LandSlope'])

df['Neighborhood'] = LabelEncoder().fit_transform(df['Neighborhood'])

df['Condition1'] = LabelEncoder().fit_transform(df['Condition1'])

df['Condition2'] = LabelEncoder().fit_transform(df['Condition2'])

df['BldgType'] = LabelEncoder().fit_transform(df['BldgType'])

df['RoofMatl'] = LabelEncoder().fit_transform(df['RoofMatl'])

df['RoofStyle'] = LabelEncoder().fit_transform(df['RoofStyle'])

df['ExterQual'] = LabelEncoder().fit_transform(df['ExterQual'])

df['ExterCond'] = LabelEncoder().fit_transform(df['ExterCond'])

df['Foundation'] = LabelEncoder().fit_transform(df['Foundation'])

df['Heating'] = LabelEncoder().fit_transform(df['Heating'])

df['HeatingQC'] = LabelEncoder().fit_transform(df['HeatingQC'])

df['CentralAir'] = LabelEncoder().fit_transform(df['CentralAir'])

df['PavedDrive'] = LabelEncoder().fit_transform(df['PavedDrive'])

df['SaleCondition'] = LabelEncoder().fit_transform(df['SaleCondition'])



test_X=df[['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 

             'YearRemodAdd', 'RoofMatl', 'Street', 'LotShape', 'LandContour', 'LotConfig', 

             'RoofStyle', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 

             'BldgType', 'ExterQual', 'ExterCond', 'Foundation','MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 

             'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 

             'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',  

             'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 

             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PavedDrive', 'SaleCondition', 

             'YrSold']]

test_y=test_X.values.reshape(-1,53)



predicted_price=xgb_reg.predict(test_X)
my_submission = pd.DataFrame({'Id': df.Id, 'SalePrice': predicted_price})

my_submission.to_csv('submissionXGBoost.csv', index=False)



# Your submission scored 0.13928, which is not an improvement of your best score. Keep trying!