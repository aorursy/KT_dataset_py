# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *









# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

s = (home_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

all_cols = list(home_data.columns)

all_cols.remove('SalePrice')

cols_with_missing = [col for col in home_data.columns

                     if home_data[col].isnull().any()]

print(cols_with_missing)

features_all = [x for x in all_cols if x not in object_cols]

#features_all.remove('SalePrice')

#print(features_all)

# Create target object and call it y

home_data = home_data.drop(home_data[(home_data['GrLivArea'] > 4000) & (home_data['SalePrice'] < 300000)].index)

home_data["SalePrice"] = np.log1p(home_data["SalePrice"])

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',

            'OverallQual','OverallCond','PoolArea','YearRemodAdd','MasVnrArea','BsmtFinSF1','TotalBsmtSF','LowQualFinSF',

           'GrLivArea','BsmtFullBath','HalfBath','KitchenAbvGr','Fireplaces','GarageCars','GarageArea','WoodDeckSF']

#Below code from https://www.kaggle.com/sagarmainkar/sagar

all_data = home_data[all_cols]





all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].mean())



for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 

            'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType',

           'MSSubClass'):

    all_data[col] = all_data[col].fillna('None')



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','GarageYrBlt', 

            'GarageArea', 'GarageCars','MasVnrArea'):

    all_data[col] = all_data[col].fillna(0)



for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 

            'Utilities'):

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])





# MSSubClass is the building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



# Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)



# Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',

            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',

            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',

            'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(all_data[c].values))

    all_data[c] = lbl.transform(list(all_data[c].values))



# control shape

print('Shape all_data: {}'.format(all_data.shape))

all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']



s = (all_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)

feat_all2 = list(all_data.columns)

newfeat = [x for x in feat_all2 if x not in object_cols]

#X = X.fillna(X.mean())

X = all_data[newfeat]

#print(X.head)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators=500, random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

from xgboost import XGBRegressor

#XGB params from https://www.kaggle.com/itslek/stack-blend-lrs-xgb-lgb-house-prices-k-v17

xgboost_on_full_data = XGBRegressor(learning_rate=0.01, n_estimators=3500,

                       max_depth=3, min_child_weight=0,

                       gamma=0, subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear', nthread=-1,

                       scale_pos_weight=1, seed=27,

                       reg_alpha=0.00006)

#XGBRegressor(n_estimators=300, learning_rate=0.05) #tried n_est from 200 to 1000 - 300 best

xgboost_on_full_data.fit(X, y) #lr = 0.05 good, lr=0.01 worse with 300
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



test_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].mean())



for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 

            'GarageCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MasVnrType',

           'MSSubClass'):

    test_data[col] = test_data[col].fillna('None')



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','GarageYrBlt', 

            'GarageArea', 'GarageCars','MasVnrArea'):

    test_data[col] = test_data[col].fillna(0)



for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional', 

            'Utilities'):

    test_data[col] = test_data[col].fillna(test_data[col].mode()[0])





# MSSubClass is the building class

test_data['MSSubClass'] = test_data['MSSubClass'].apply(str)



# Changing OverallCond into a categorical variable

test_data['OverallCond'] = test_data['OverallCond'].astype(str)



# Year and month sold are transformed into categorical features.

test_data['YrSold'] = test_data['YrSold'].astype(str)

test_data['MoSold'] = test_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',

            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',

            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',

            'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(test_data[c].values))

    test_data[c] = lbl.transform(list(test_data[c].values))



# control shape

#print('Shape all_data: {}'.format(all_data.shape))

test_data['TotalSF'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[newfeat]

test_X = test_X.fillna(test_X.mean())

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)

test_preds_xg = xgboost_on_full_data.predict(test_X)

test_preds_xg_exp = np.expm1(test_preds_xg)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds_xg_exp})

output.to_csv('submission.csv', index=False)