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
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
train
train.describe()
train.head()
train.columns
features = ['MSSubClass', 'LotFrontage', 'LotArea', 'Street',

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

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'

       ]
X = train[features]
X.columns
y = train.SalePrice
X.describe()
y.describe()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X,y, train_size=0.8, test_size=0.2, random_state=0)
cols_missing = [col for col in X_train.columns

               if X_train[col].isnull().any()]

cols_missing
X_train = X_train.drop(cols_missing, axis=1)

X_val = X_val.drop(cols_missing, axis=1)
X_train.describe()
X_train.columns
X_val.describe()
cols_missing = [col for col in X_train.columns

               if X_train[col].isnull().any()]

cols_missing
# Cardinality 
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)

print('Categorical Variables:')

print(object_cols)
low_cardinality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() <= 5 and 

                        X_train[cname].dtype == "object"]
low_cardinality_cols
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
numerical_cols
my_cols = low_cardinality_cols + numerical_cols

X_train = X_train[my_cols].copy()

X_val = X_val[my_cols].copy()
s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)

print('Categorical Variables:')

print(object_cols)
X_train.describe()
X_val.describe()
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_val = X_val.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_val[col] = label_encoder.transform(X_val[col])
label_X_train
label_X_val.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor()

forest_model.fit(label_X_train, y_train)

mel_prebs = forest_model.predict(label_X_val)

print(mean_absolute_error(mel_prebs, y_val))
def get_mae(max_leaf_nodes, label_X_train, label_X_val, y_train, y_val):

    model = RandomForestRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)

    model.fit(label_X_train, y_train)

    preds_val = model.predict(label_X_val)

    mae = mean_absolute_error(preds_val, y_val)

    return mae
candidate_max_leaf_nodes = [5, 25, 35, 50, 51, 52, 53, 60, 100, 250, 500, 600, 605, 610, 800, 900]

for max_leaf_nodes in candidate_max_leaf_nodes:

    x = get_mae(max_leaf_nodes, label_X_train, label_X_val, y_train, y_val)

    print('mae', max_leaf_nodes, x)
my_model = RandomForestRegressor(max_leaf_nodes = 900, random_state=0)

my_model.fit(label_X_train, y_train)

mel_prebs = my_model.predict(label_X_val)

print(mean_absolute_error(mel_prebs, y_val))
# Test Data
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
test
test.describe()
X_test = test[features]
X_test.describe()
cols_missing_test = [col for col in X_test.columns

               if X_test[col].isnull().any()]

cols_missing_test
X_test = X_test.drop(cols_missing_test, axis=1)
X_test.describe()
X_test.columns == 'MSZoning'
i = (X_test.dtypes == 'object')

object_cols_test = list(i[i].index)

print('Categorical Variables:')

print(object_cols_test)
low_cardinality_cols_test  = [cname for cname in X_test.columns if X_test[cname].nunique() < 50000 and 

                        X_test[cname].dtype == "object"]
low_cardinality_cols_test
numerical_cols_test = [cname for cname in X_test.columns if X_test[cname].dtype in ['int64', 'float64']]
numerical_cols_test
my_cols_test = low_cardinality_cols_test + numerical_cols_test

X_test = X_test[my_cols_test].copy()

from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_test = X_test.copy()





# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in low_cardinality_cols_test:

    label_X_test[col] = label_encoder.fit_transform(X_test[col])

    
label_X_test
X_train.columns
predicted_prizes = my_model.predict(label_X_test)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prizes})

my_submission.to_csv('submission.csv', index=False)