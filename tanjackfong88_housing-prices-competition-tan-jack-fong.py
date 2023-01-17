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
# Importing necessary packages

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import xgboost as xgb

import warnings



# Removing future warning messages

warnings.simplefilter(action="ignore", category=FutureWarning)



# Path of files

X_full = pd.read_csv(("/kaggle/input/home-data-for-ml-course/train.csv"))

X_test_full = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2, random_state=0)



# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                  X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)



# Define model (base model)

xgb_model = XGBRegressor(random_state=0)



# Fit the model

xgb_model.fit(X_train, y_train)



# Get predictions

predictions = xgb_model.predict(X_valid)



# Calculate MAE

mae_1 = mean_absolute_error(predictions, y_valid)

print("Mean Absolute Error for Model 1:" , mae_1)

import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label=y_train)



# Params that will be tuned

params = {"max_depth":6, "min_child_weight":1, "eta":0.05, "subsample":1, "colsample_bytree":1,

         # Other parameters

         "objective":"reg:squarederror", "n_estimators":1000, "num_boost_round":999}
# Tuning both max_depth/min_child_weight to find a good trade-off between model bias and variance

gridsearch_params = [(max_depth, min_child_weight) 

                     for max_depth in range(9, 12) 

                     for min_child_weight in range(5, 8)]



# Defining initial best params and MAE

min_mae = float("Inf")

best_params = None

for max_depth, min_child_weight in gridsearch_params:

    print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

    

    # Update the parameters

    params["max_depth"] = max_depth

    params["min_child_weight"] = min_child_weight

    

    # Run CV

    cv_results = xgb.cv(params, dtrain, num_boost_round=999, seed=42, nfold=5, metrics={"mae"},

                       early_stopping_rounds=10)

    

    # Update best MAE

    mean_mae = cv_results["test-mae-mean"].min()

    boost_rounds = cv_results["test-mae-mean"].argmin()

    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (max_depth, min_child_weight)

        

print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
# Update max_depth and min_child_weight parameters

params["max_depth"] = 11

params["min_child_weight"] = 5
# Tuning subsample and colsample_bytree

gridsearch_params = [(subsample, colsample) for subsample in [i/10. for i in range(7, 11)]

                    for colsample in [i/10. for i in range(7,11)]]



min_mae = float("Inf")

best_params = None



# Start with the largest value and go down to the smallest

for subsample, colsample in reversed(gridsearch_params):

    print("CV with subsample={}, colsample={}".format(subsample,colsample))

    

    # Update the parameters

    params["subsample"] = subsample

    params["colsample_bytree"] = colsample

    

    # Run CV

    cv_results = xgb.cv(params, dtrain, num_boost_round=999, seed=42, nfold=5, metrics="mae",

                        early_stopping_rounds=10)

    

    # Update best score

    mean_mae = cv_results["test-mae-mean"].min()

    boost_rounds = cv_results["test-mae-mean"].argmin()

    print("\MAE {} for {} rounds".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = (subsample, colsample)



print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))
# Update subsample and colsample_bytree parameters

params["subsample"] = 0.7

params["colsample_bytree"] = 0.7
# Tuning ETA which controls the learning rate

min_mae = float("Inf")

best_params = None



for eta in [.3, .2, .1, .05, .01, .005]:

    print("CV with eta={}".format(eta))

    

    # Update the parameter

    params["eta"]=eta

    

    # Run and time CV

    cv_results = xgb.cv(params, dtrain, num_boost_round=999, seed=42, nfold=5, metrics="mae",

                        early_stopping_rounds=10)

    

    #Update best score

    mean_mae = cv_results["test-mae-mean"].min()

    boost_rounds = cv_results["test-mae-mean"].argmin()

    print("\MAE {} for {} rounds\n".format(mean_mae, boost_rounds))

    if mean_mae < min_mae:

        min_mae = mean_mae

        best_params = eta



print("Best params: {}, MAE: {}".format(best_params, min_mae))
# Update eta parameter

params["eta"] = 0.01
# Relook at best parameters chosen and deploy on test model 

params
# Change parameters on the base XGBoost model

xgb_model_2 = XGBRegressor(objective="reg:squarederror", n_estimators=1000, eta=0.01, subsample=0.7, 

                           colsample_bytree=0.7,max_depth=11, min_child_weight=5, num_boost_round=999) 

xgb_model_2.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False )

xgb_val_predictions_2 = xgb_model_2.predict(X_valid)

xgb_val_mae_2 = mean_absolute_error(xgb_val_predictions_2, y_valid)

print("Validation MAE for XGBoost Model: {:,.0f}".format(xgb_val_mae_2))
# Relook at best parameters chosen and deploy on final model 

params
# Deploying model on test data

xgb_model_final = XGBRegressor(objective="reg:squarederror", n_estimators=1000, eta=0.01, 

                               subsample=0.7, colsample_bytree=0.7, max_depth=11, min_child_weight=5) 

xgb_model_final.fit(X_train, y_train)

test_preds = xgb_model_final.predict(X_test)
# Submitting predictions

output = pd.DataFrame({'Id': X_test.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)