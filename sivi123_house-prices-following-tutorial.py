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
train_file_path = '../input/train.csv'
test_file_path =  '../input/test.csv'
# read the data and store data in DataFrame titled melbourne_data
train_data = pd.read_csv(train_file_path) 
test_data = pd.read_csv(test_file_path)

# Use Shape to get Matrices sizes 
print ("train shape: " , train_data.shape )
print ("test shape: " , test_data.shape )



# A summary of the data in train data
train_data.describe()

pd.set_option('display.expand_frame_repr', True)
#print (train_data)

# use count to see which columns have missing data
print (train_data.count())
d = test_data.select_dtypes(exclude=['object'])

print (d.count())
# Summery of data in test 
test_data.describe()
# all columns 
print(train_data.columns)
cols_to_use = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        'SaleType', 'SaleCondition']

# moved these valuse because they are suspuctable of data leaking
#'MiscVal','Fence','MiscFeature',  'MoSold', 'YrSold'
#  'GarageYrBlt',

# using as much as data as possible removing all strings for simp;
train_X = train_data[cols_to_use].select_dtypes(exclude=['object'])
train_y = train_data.SalePrice
test_X = test_data[cols_to_use].select_dtypes(exclude=['object'])



#remove LotFrontage tage
train_X = train_X.drop(columns='LotFrontage' )
test_X = test_X.drop(columns='LotFrontage')

print (train_X.count())

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

imputed_X_train_plus = train_X.copy()
imputed_X_test_plus = test_X.copy()

cols_with_missing = (col for col in train_X.columns 
                                 if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

my_pipeline = make_pipeline(my_imputer, XGBRegressor(n_estimators=1000, learning_rate=0.011))
#my_pipeline = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())


my_pipeline.fit(imputed_X_train_plus, train_y)
predictions = my_pipeline.predict(imputed_X_test_plus)


#crossvalidation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, train_X, train_y, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))


my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_sivan.csv', index=False)
