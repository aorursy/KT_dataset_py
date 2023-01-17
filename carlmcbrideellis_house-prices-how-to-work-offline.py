#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a minimal script to perform a regression on the kaggle 

# 'House Prices' data set.

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd

import numpy   as np



#===========================================================================

# read in the competition data 

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# also, read in the 'solution' data 

# Note: you either need to use "+ Add data" to include this file if you are woking on kaggle,

# or download it and store it locally if you are completely offline

#===========================================================================

solution   = pd.read_csv('../input/house-prices-advanced-regression-solution-file/solution.csv')

y_true     = solution["SalePrice"]

                         

#===========================================================================

# select some features of interest

#===========================================================================

features = ['OverallQual', 'GrLivArea', 'GarageCars',  'TotalBsmtSF']



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]



#===========================================================================

# essential preprocessing: imputation; substitute any 'NaN' with mean value

#===========================================================================

X_train      = X_train.fillna(X_train.mean())

final_X_test = final_X_test.fillna(final_X_test.mean())



#===========================================================================

# perform the regression and then the fit

#===========================================================================

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, max_depth=7)

regressor.fit(X_train, y_train)



#===========================================================================

# use the model to predict the prices for the test data

#===========================================================================

y_pred = regressor.predict(final_X_test)



#===========================================================================

# compare your predictions with the 'solution' using the 

# root of the mean_squared_log_error

#===========================================================================

from sklearn.metrics import mean_squared_log_error

RMSLE = np.sqrt( mean_squared_log_error(y_true, y_pred) )

print("The score is %.5f" % RMSLE )
#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":y_pred})

output.to_csv('submission.csv', index=False)