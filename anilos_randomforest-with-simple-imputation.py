# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_file_path = '../input/home-data-for-ml-course/train.csv'
train_data = pd.read_csv(train_file_path)
train_data.columns
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

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

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']
X = train_data[features]
X.columns
y = train_data.SalePrice
y.describe()
X.describe()
X.head()
X = train_data.select_dtypes(exclude=['object'])
X.columns
feature_train = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']
X = train_data[feature_train]
X.describe()
y.describe()
from sklearn.model_selection import train_test_split



val_X, train_X, val_y, train_y = train_test_split(X,y, random_state=0)
val_X.describe()
train_X.describe()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def score_data_set(val_X, train_X, val_y, train_y):

    model = RandomForestRegressor(n_estimators = 10, random_state = 1)

    model.fit(train_X, train_y)

    preds = model.predict(val_X)

    return mean_absolute_error(preds, val_y)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))

imputed_val_X = pd.DataFrame(my_imputer.transform(val_X))

# pd.DataFrame(my_imputer.transform(X_valid))



imputed_train_X.columns = train_X.columns 

imputed_val_X.columns = val_X.columns



imputed_train_X.describe()
imputed_val_X.describe()
forest_model = RandomForestRegressor()



forest_model.fit(imputed_train_X, train_y)

mel_prebs = forest_model.predict(imputed_val_X)

print(mean_absolute_error(mel_prebs, val_y))
# Underfitting and Overfitting
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



def get_mae(max_leaf_nodes, imputed_val_X, imputed_train_X, val_y, train_y):

    model = RandomForestRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)

    model.fit(imputed_train_X, train_y)

    preds_val = model.predict(imputed_val_X)

    mae = mean_absolute_error(preds_val, val_y)

    return(mae)
candidate_max_leaf_nodes = [5, 25, 35, 50, 51, 52, 53, 60, 100, 250, 500, 600]



for max_leaf_nodes in candidate_max_leaf_nodes:

    x = get_mae(max_leaf_nodes, imputed_val_X, imputed_train_X, val_y, train_y)

    print('mae', max_leaf_nodes, x)

my_model = RandomForestRegressor(max_leaf_nodes = 600, random_state=1)

my_model.fit(imputed_train_X, train_y)

mel_prebs = my_model.predict(imputed_val_X)

print(mean_absolute_error(mel_prebs, val_y))
# Test Data
test_data_path = '../input/home-data-for-ml-course/test.csv'
test_data = pd.read_csv(test_data_path)
test_data.describe()
X_test.columns
X_test = test_data[feature_train]
X_test.describe()
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

imputed_X_test.columns = X_test.columns
predicted_prizes = my_model.predict(imputed_X_test)
predicted_prizes
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prizes})

my_submission.to_csv('submission.csv', index=False)