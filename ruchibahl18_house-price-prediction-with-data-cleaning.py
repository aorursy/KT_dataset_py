# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.imputation import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
data = train.append(test,sort=True)
#data = train
data = data.drop(['SalePrice'], axis = 1)
data.describe()
data.head()
data.columns
data.isnull().values.any()
data.isnull().sum()
data.isna()
nullableColumns = data.columns[data.isna().any()==True].tolist()
nullableColumns
data = train
data = data.drop(nullableColumns, axis=1)
data = data.drop(['SalePrice'], axis = 1)
data.head()
data.dtypes
Y = train['SalePrice']
Y.shape
X = data
X.shape
#my_imputer = Imputer()
le = preprocessing.LabelEncoder()
X.head()
#X['MSZoning'] = le.fit_transform(X['MSZoning'])
X['LotShape'] = le.fit_transform(X['LotShape'])
X['LandContour'] = le.fit_transform(X['LandContour'])
X['LotConfig'] = le.fit_transform(X['LotConfig'])
X['LandSlope'] = le.fit_transform(X['LandSlope'])
X['Neighborhood'] = le.fit_transform(X['Neighborhood'])
X['Condition1'] = le.fit_transform(X['Condition1'])
X['Condition2'] = le.fit_transform(X['Condition2'])
X['BldgType'] = le.fit_transform(X['BldgType'])
X['HouseStyle'] = le.fit_transform(X['HouseStyle'])
X['RoofStyle'] = le.fit_transform(X['RoofStyle'])
X['RoofMatl'] = le.fit_transform(X['RoofMatl'])
#X['Exterior1st'] = le.fit_transform(X['Exterior1st'])

#X['Exterior2nd'] = le.fit_transform(X['Exterior2nd'])
#X['MasVnrType'] = le.fit_transform(X['MasVnrType'])
X['ExterQual'] = le.fit_transform(X['ExterQual'])
X['ExterCond'] = le.fit_transform(X['ExterCond'])
X['Foundation'] = le.fit_transform(X['Foundation'])

#X['BsmtQual'] = le.fit_transform(X['BsmtQual'])
#X['BsmtCond'] = le.fit_transform(X['BsmtCond'])
# X['BsmtExposure'] = le.fit_transform(X['BsmtExposure'])
# X['BsmtFinType1'] = le.fit_transform(X['BsmtFinType1'])
# X['BsmtFinType2'] = le.fit_transform(X['BsmtFinType2'])
X['Heating'] = le.fit_transform(X['Heating'])
X['HeatingQC'] = le.fit_transform(X['HeatingQC'])

X['CentralAir'] = le.fit_transform(X['CentralAir'])
# X['Electrical'] = le.fit_transform(X['Electrical'])
#X['KitchenQual'] = le.fit_transform(X['KitchenQual'])
#X['Functional'] = le.fit_transform(X['Functional'])

#X['FireplaceQu'] = le.fit_transform(X['FireplaceQu'])
#X['GarageType'] = le.fit_transform(X['GarageType'])
#X['GarageFinish'] = le.fit_transform(X['GarageFinish'])
#X['GarageQual'] = le.fit_transform(X['GarageQual'])
#X['GarageCond'] = le.fit_transform(X['GarageCond'])
X['PavedDrive'] = le.fit_transform(X['PavedDrive'])
X['Street'] = le.fit_transform(X['Street'])

#X['Utilities'] = le.fit_transform(X['Utilities'])
#X['SaleType'] = le.fit_transform(X['SaleType'])
X['SaleCondition'] = le.fit_transform(X['SaleCondition'])
X.head()
train_X, val_X, train_y, val_y = train_test_split(X, Y, random_state=0)
#my_pipeline = make_pipeline(my_imputer,RandomForestRegressor())
my_pipeline = make_pipeline(RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(val_X)
mean_absolute_error(val_y, predictions)
scores = cross_val_score(my_pipeline, X, Y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
test.head()
test.isnull().values.any()
test = test.drop(nullableColumns, axis=1)
#test['MSZoning'] = le.fit_transform(test['MSZoning'])
test['LotShape'] = le.fit_transform(test['LotShape'])
test['LandContour'] = le.fit_transform(test['LandContour'])
test['LotConfig'] = le.fit_transform(test['LotConfig'])
test['LandSlope'] = le.fit_transform(test['LandSlope'])
test['Neighborhood'] = le.fit_transform(test['Neighborhood'])
test['Condition1'] = le.fit_transform(test['Condition1'])
test['Condition2'] = le.fit_transform(test['Condition2'])
test['BldgType'] = le.fit_transform(test['BldgType'])
test['HouseStyle'] = le.fit_transform(test['HouseStyle'])
test['RoofStyle'] = le.fit_transform(test['RoofStyle'])
test['RoofMatl'] = le.fit_transform(test['RoofMatl'])
#test['Exterior1st'] = le.fit_transform(test['Exterior1st'])

#test['Exterior2nd'] = le.fit_transform(test['Exterior2nd'])
#test['MasVnrType'] = le.fit_transform(test['MasVnrType'])
test['ExterQual'] = le.fit_transform(test['ExterQual'])
test['ExterCond'] = le.fit_transform(test['ExterCond'])
test['Foundation'] = le.fit_transform(test['Foundation'])

#test['BsmtQual'] = le.fit_transform(test['BsmtQual'])
#test['BsmtCond'] = le.fit_transform(test['BsmtCond'])
# test['BsmtExposure'] = le.fit_transform(test['BsmtExposure'])
# test['BsmtFinType1'] = le.fit_transform(test['BsmtFinType1'])
# test['BsmtFinType2'] = le.fit_transform(test['BsmtFinType2'])
test['Heating'] = le.fit_transform(test['Heating'])
test['HeatingQC'] = le.fit_transform(test['HeatingQC'])

test['CentralAir'] = le.fit_transform(test['CentralAir'])
# test['Electrical'] = le.fit_transform(test['Electrical'])
#test['KitchenQual'] = le.fit_transform(test['KitchenQual'])
#test['Functional'] = le.fit_transform(test['Functional'])

#test['FireplaceQu'] = le.fit_transform(test['FireplaceQu'])
#test['GarageType'] = le.fit_transform(test['GarageType'])
#test['GarageFinish'] = le.fit_transform(test['GarageFinish'])
#test['GarageQual'] = le.fit_transform(test['GarageQual'])
#test['GarageCond'] = le.fit_transform(test['GarageCond'])
test['PavedDrive'] = le.fit_transform(test['PavedDrive'])
test['Street'] = le.fit_transform(test['Street'])

#test['Utilities'] = le.fit_transform(test['Utilities'])
#test['SaleType'] = le.fit_transform(test['SaleType'])
test['SaleCondition'] = le.fit_transform(test['SaleCondition'])
test.head()
predicted_prices = my_pipeline.predict(test)
my_submission = pd.DataFrame({'Id':test['Id'],'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
