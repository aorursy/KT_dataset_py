# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
#features = lists(home_data)
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Any results you write to the current directory are saved as output.
# one hot encoding
OHE_features =  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
                        ,'LotShape', 'Neighborhood', 'Condition1','HouseStyle']
OHE_y = home_data.SalePrice
OHE_X = home_data[OHE_features]
#***
future_X = pd.get_dummies(OHE_X)
#***
OHE_train_X, OHE_val_X, OHE_train_y, OHE_val_y = train_test_split(OHE_X, OHE_y, random_state=1)
one_hot_encoded_training_predictors = pd.get_dummies(OHE_train_X)
one_hot_encoded_test_predictors = pd.get_dummies(OHE_val_X)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
# use Imputation
from sklearn.impute import SimpleImputer
myImputer = SimpleImputer()
imputed_train_X = myImputer.fit_transform(train_X)
imputed_val_X = myImputer.transform(val_X)
# using decision_tree_regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
# Specify Model
iowa_model = DecisionTreeRegressor(max_leaf_nodes = 70 ,random_state=1)
iowa_model_Two = DecisionTreeRegressor(max_leaf_nodes = 70 ,random_state=1)
#********************************************************************************

# Fit Model
iowa_model.fit(train_X, train_y)
# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE without Imputation: {:,.0f}".format(val_mae))
#********************************************************************************

#imputed Model
iowa_model_Two.fit(imputed_train_X, train_y)
# Make validation predictions and calculate mean absolute error
imputed_val_predictions = iowa_model_Two.predict(imputed_val_X)
imputed_val_mae = mean_absolute_error(imputed_val_predictions, val_y)
print("Validation MAE with    Imputation: {:,.0f}".format(imputed_val_mae))
# using random_forest_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(random_state=1)
rf_model_Two = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
#*****************************************************************

#imputed Model
rf_model_Two.fit(imputed_train_X, train_y)
# Make validation predictions and calculate mean absolute error
imputed_val_predictions = rf_model_Two.predict(imputed_val_X)
imputed_val_mae = mean_absolute_error(imputed_val_predictions, val_y)
print("Validation MAE with    Imputation: {:,.0f}".format(imputed_val_mae))
# use XGBoost Regressor
from xgboost import XGBRegressor

xg_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.08)
xg_model.fit(train_X, train_y, early_stopping_rounds = 10, eval_set=[(val_X,val_y)], verbose = False)
imputed_xg_val_predictions = xg_model.predict(val_X)
imputed_xg_val_mae = mean_absolute_error(imputed_xg_val_predictions, val_y)
print("Validation MAE for Xgboost: {:,.0f}".format(imputed_xg_val_mae))

# with one_hot_encoding
OHE_xg_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.08)
OHE_xg_model.fit(final_train, OHE_train_y, early_stopping_rounds = 10, eval_set=[(final_test, OHE_val_y)],
                 verbose = False)
OHE_val_predictions = OHE_xg_model.predict(final_test)
OHE_val_mae = mean_absolute_error(OHE_val_predictions, OHE_val_y)
print("Validation MAE with OHE: {:,.0f}".format(OHE_val_mae))

# cross validation
my_pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = 500, learning_rate = 0.05))
scores = cross_val_score(my_pipeline, future_X, y, scoring='neg_mean_absolute_error')
print('Mean Absolute Error %2f' %(-1 * scores.mean()))
# make prediction and submit
from sklearn.pipeline import make_pipeline
# path to file you will use for predictions
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)
xg_model_sub = XGBRegressor(n_estimators = 1000, learning_rate = 0.08)
xg_model_sub.fit(X, y, verbose = False)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X,y)
OHE_test_X = test_data[OHE_features]

#****************************************************************
OHE_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.08)
my_pipeline.fit(future_X,y)
final_test_data = pd.get_dummies(OHE_test_X)
tmp,ordered_final_test = future_X.align(final_test_data,join='left',axis=1)
#****************************************************************
# make predictions which we will submit. 
test_preds = my_pipeline.predict(ordered_final_test)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
#partial dependence plot with GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
my_model = GradientBoostingRegressor()
my_model.fit(X,y)
my_plots = plot_partial_dependence(my_model, features = [0,2], X=X, feature_names=
                                   ['LotArea', 'GarageCars', 'BuildingArea'],grid_resolution=10)
