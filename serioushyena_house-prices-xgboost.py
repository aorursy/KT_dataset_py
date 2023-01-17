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
from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf

from keras import Sequential

from keras.layers import Dense

from keras import utils



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import math



import xgboost as xgb
#THIS STORES THE HOUSE ID'S FOR LATER SUBMISSION

test_id = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', usecols=['Id'])

test_id = test_id.astype('int32').round(0)
########## READ IN THE DATA AND ASSIGN x,Y #################################

X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', usecols=['LotArea', '1stFlrSF',

                    'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',

                    'OpenPorchSF', 'Neighborhood', 'LotFrontage', 

                    'OverallQual', 'OverallCond', 'ExterQual', 'FullBath',

                    'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

                    'GarageType', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                    'EnclosedPorch', 'ScreenPorch', 'SaleType'])
X.head(5)
X.shape
y = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', usecols=['SalePrice'])
y.head(5)
y.shape
####################3  DEAL WITH NANS ################################3

X['LotFrontage'].fillna(X['LotFrontage'].mean)
############ READ IN THE TEST DATA SET ########################################

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', usecols=['LotArea', '1stFlrSF',

                    'TotalBsmtSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',

                    'OpenPorchSF', 'Neighborhood', 'LotFrontage', 

                    'OverallQual', 'OverallCond', 'ExterQual', 'FullBath',

                    'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

                    'GarageType', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',

                    'EnclosedPorch', 'ScreenPorch', 'SaleType'])
# CHECK FOR NAN

X['SaleType'].isna().value_counts()

#X['SaleType'].fillna('Attchd', inplace=True)

test['SaleType'].isna().value_counts()

test['SaleType'].fillna('Oth', inplace=True)



X['GarageType'].isna().value_counts()

X['GarageType'].fillna('Attchd', inplace=True)

test['GarageType'].isna().value_counts()

test['GarageType'].fillna('Attchd', inplace=True)
####################3  DEAL WITH NANS ################################3

test['LotFrontage'].fillna(test['LotFrontage'].mean)
####################### DROP ROWS TO MAKE THE SHAPES THE SAME #####################

#I can see that train has one more row than test so let's drop the last rof of 

# test in order to keep the same shape to avoid problems with shape mis-match

X.shape

y.shape

test.shape



X.drop(X.tail(1).index, inplace=True)

y.drop(y.tail(1).index, inplace=True)
############ EDA ON THE TRAIN DATA SET ############################################################

'''

I know I should write a function to streamline this and reduce code but just wanted to get a working model first

'''



from sklearn.preprocessing import OneHotEncoder

# OneHotEncoder-sex column

X_neighborhood = OneHotEncoder().fit_transform(X['Neighborhood'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

X[['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',

       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',

       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',

       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',

       'Blueste']] = pd.DataFrame(X_neighborhood, index = X.index)

X.drop(['Neighborhood'], axis=1, inplace=True)



test_neighborhood = OneHotEncoder().fit_transform(test['Neighborhood'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

test[['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',

       'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes',

       'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert',

       'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU',

       'Blueste']] = pd.DataFrame(test_neighborhood, index = test.index)

test.drop(['Neighborhood'], axis=1, inplace=True)
X_xter_qual = OneHotEncoder().fit_transform(X['ExterQual'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

X[['TA', 'Gd', 'Ex', 'Fa']] = pd.DataFrame(X_xter_qual, index = X.index)

X.drop(['ExterQual'], axis=1, inplace=True)



test_xter_qual = OneHotEncoder().fit_transform(test['ExterQual'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

test[['TA', 'Gd', 'Ex', 'Fa']] = pd.DataFrame(test_xter_qual, index = test.index)

test.drop(['ExterQual'], axis=1, inplace=True)
X_garage = OneHotEncoder().fit_transform(X['GarageType'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

X[['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types']] = pd.DataFrame(X_garage, index = X.index)

X.drop(['GarageType'], axis=1, inplace=True)



test_garage = OneHotEncoder().fit_transform(test['GarageType'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

test[['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types']] = pd.DataFrame(test_garage, index = test.index)

test.drop(['GarageType'], axis=1, inplace=True)
X_sale_type = OneHotEncoder().fit_transform(X['SaleType'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

X[['WD', 'COD', 'New', 'ConLD', 'Oth', 'Con', 'ConLw', 'ConLI', 'CWD']] = pd.DataFrame(X_sale_type, index = X.index)

X.drop(['SaleType'], axis=1, inplace=True)



test_sale_type = OneHotEncoder().fit_transform(test['SaleType'].values.reshape(-1, 1)).toarray()

# Appending with male, female columns

test[['WD', 'COD', 'New', 'ConLD', 'Oth', 'Con', 'ConLw', 'ConLI', 'CWD']] = pd.DataFrame(test_sale_type, index = test.index)

test.drop(['SaleType'], axis=1, inplace=True)



########################## FEATURE SCALING ################################

print(X.var())

print(test.var())

type(test)

type(X)



scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)



test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)

print(test_scaled.var())
#################### BUILD XGBOOST MODEL BASELEARNER ###################################3

#now build the xgboost learner





# by increasing the n_estimators from 40 to 80 i saw an increase in accuracy

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=80, seed=123)



xg_reg.fit(X_scaled, y)



preds = xg_reg.predict(test_scaled)

preds[:5]
################### TUNED MODEL QUALITY ###############################

# Create the DMatrix: housing_dmatrix

housing_dmatrix = xgb.DMatrix(data=X, label=y)



# Create the parameter dictionary: params

tuned_params = {"objective":"reg:squarederror", 'colsample_bytree':0.3,

              'learning_rate':0.1, "max_depth":5}



# Perform cross-validation: cv_results

tuned_cv_results = xgb.cv(dtrain=housing_dmatrix, params=tuned_params, 

              nfold=4, num_boost_round=200, metrics='rmse', 

              as_pandas=True, seed=123)



# Print cv_results

print(tuned_cv_results.tail(1))

tuned_cv_results.shape
# Extract and print final boosting round metric

print((tuned_cv_results["test-rmse-mean"]).tail(1))
####### CONCATENATE house_ids TO PREDICTIONS ########################

predictions = pd.DataFrame(preds)

predictions.rename(columns={0:'SalePrice'}, inplace=True)

predictions = predictions.astype('int32')

test_id = test_id.astype('int32')

submission = pd.concat([test_id, predictions], axis=1)





submission.to_csv('submission.csv', index=False)

subm = pd.read_csv('submission.csv')

subm



!kaggle competitions submit -c house-prices-advanced-regression-techniques -f ./submission.csv -m "Message"
'''

This ia a quick and dirty xgboost model. I am still learning about xgboost. This is my first time using it so I am in the process of exploring parameter tuning

to try and get the most accuracy from the model. I know it's not well commented, I apologise for that but just wanted to get something going that I can narrate as I 

learn more and see the results of my experiments. Thanks for understanding

'''