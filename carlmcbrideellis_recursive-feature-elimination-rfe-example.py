#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a simple script to perform recursive feature elimination on 

# the kaggle 'House Prices' data set using the scikit-learn RFE

# Carl McBride Ellis (2.V.2020)

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features to rank. These are all 'integer' fields for today.

#===========================================================================

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 

        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 

        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 

        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 

        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 

        'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

        'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]



#===========================================================================

# simple preprocessing: imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

final_X_test = final_X_test.fillna(final_X_test.mean())



#===========================================================================

# set up our regressor. Today we shall be using the random forest

#===========================================================================

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, max_depth=10)



#===========================================================================

# perform a scikit-learn Recursive Feature Elimination (RFE)

#===========================================================================

from sklearn.feature_selection import RFE

# here we want only one final feature, we do this to produce a ranking

n_features_to_select = 1

rfe = RFE(regressor, n_features_to_select)

rfe.fit(X_train, y_train)



#===========================================================================

# now print out the features in order of ranking

#===========================================================================

from operator import itemgetter

for x, y in (sorted(zip(rfe.ranking_ , features), key=itemgetter(0))):

    print(x, y)



#===========================================================================

# ok, this time let's choose the top 8 featues and use them for the model

#===========================================================================

n_features_to_select = 8

rfe = RFE(regressor, n_features_to_select)

rfe.fit(X_train, y_train)



#===========================================================================

# use the model to predict the prices for the test data

#===========================================================================

predictions = rfe.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)