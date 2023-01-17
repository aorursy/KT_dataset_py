#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a minimal script to perform a regression on the kaggle 

# 'House Prices' data set using the XGBoost Python API 

# Carl McBride Ellis (11.IV.2020)

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas  as pd

import xgboost as xgb



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

#===========================================================================

features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,

            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]



#===========================================================================

#===========================================================================

X_train       = train_data[features]

y_train       = train_data["SalePrice"]

final_X_test  = test_data[features]



#===========================================================================

# XGBoost regression: 

# Parameters: 

# n_estimators  "Number of gradient boosted trees. Equivalent to number 

#                of boosting rounds."

# learning_rate "Boosting learning rate (xgb’s “eta”)"

# max_depth     "Maximum depth of a tree. Increasing this value will make 

#                the model more complex and more likely to overfit." 

#===========================================================================

regressor=xgb.XGBRegressor(n_estimators  = 750,

                           learning_rate = 0.02,

                           max_depth     = 3)

regressor.fit(X_train, y_train)



#===========================================================================

# To use early_stopping_rounds: 

# "Validation metric needs to improve at least once in every 

# early_stopping_rounds round(s) to continue training."

#===========================================================================

# perform a test/train split 

#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size = 0.5,random_state= 0)

#regressor = xgb.XGBRegressor(n_estimators=750, learning_rate=0.02)

#regressor.fit(X_train, y_train, early_stopping_rounds=6, eval_set=[(X_test, y_test)], verbose=False)



#===========================================================================

# use the model to predict the prices for the test data

#===========================================================================

predictions = regressor.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)