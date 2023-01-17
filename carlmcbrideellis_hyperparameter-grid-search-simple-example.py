#!/usr/bin/python3

# coding=utf-8

#===========================================================================

# This is a simple script to perform a regression on the kaggle

# 'House Prices' data set using a grid search, in conjunction with a

# random forest regressor

# Carl McBride Ellis (1.V.2020)

#===========================================================================

#===========================================================================

# load up the libraries

#===========================================================================

import pandas as pd

import numpy  as np 



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_data  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

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

# hyperparameter grid search using scikit-learn GridSearchCV

# we use the default 5-fold cross validation

#===========================================================================

from sklearn.model_selection import GridSearchCV

# we use the random forest regressor

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()

gs = GridSearchCV(cv=5, error_score=np.nan, estimator=regressor,

# dictionaries containing values to try for the parameters

param_grid={'max_depth'   : [ 2,  5,  7, 10],

            'n_estimators': [20, 30, 50, 75]})

gs.fit(X_train, y_train)



# grid search has finished, now echo the results to the screen

print("The best parameters are ",gs.best_params_)

the_best_parameters = gs.best_params_



#===========================================================================

# perform the regression 

#===========================================================================

regressor = RandomForestRegressor(

                     n_estimators = the_best_parameters["n_estimators"],

                     max_depth    = the_best_parameters["max_depth"])

regressor.fit(X_train, y_train)



#===========================================================================

# use the model to predict the prices for the test data

#===========================================================================

predictions = regressor.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({"Id":test_data.Id, "SalePrice":predictions})

output.to_csv('submission.csv', index=False)