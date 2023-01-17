# Code you have previously used to load data

import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from learntools.core import *

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

train_data = pd.read_csv(iowa_file_path)



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# Use one hot encoding

one_hot_encoded_train_data = pd.get_dummies(train_data)

one_hot_encoded_test_data = pd.get_dummies(test_data)

final_train_data, final_test_data = one_hot_encoded_train_data.align(one_hot_encoded_test_data,

                                                                    join = 'left',

                                                                    axis = 1)



# Chosse columns to use

y = final_train_data.SalePrice

X = final_train_data.drop(['SalePrice'], axis=1)

test_X = final_test_data.drop(['SalePrice'], axis=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2)
# Find columns including missing value

# missing_val_count_by_col = home_data.isnull().sum()

# print(missing_val_count_by_col[missing_val_count_by_col > 0])



# Make pipelines

# rf_pipelines = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state = 1))

xgb_pipelines = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = 1000, learning_rate = 0.03))

xgb_pipelines.fit(X, y)

pred_y = xgb_pipelines.predict(test_X)



# # Use cross validation and find the better model

# # rf_score = cross_val_score(rf_pipelines,X, y, cv=5, scoring='neg_mean_absolute_error')

# xgb_score = cross_val_score(xgb_pipelines, X, y, cv=5, scoring='neg_mean_absolute_error')



# # print('rf_score: ' + str(rf_score.mean()))

# print('xgb_score: ' + str(xgb_score.mean()))



# # Find the best hyper parameter

# for i in [100, 200, 300, 400, 500, 1000]:

#     for j in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:

#         xgb_pipelines = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators = i, learning_rate = j))

#         xgb_score = cross_val_score(xgb_pipelines, X, y, cv=5, scoring='neg_mean_absolute_error')

#         print('xgb_score with n_estimators=' + str(i) + 'and learning_rate = ' + str(j) +' : ' + str(xgb_score.mean()))
# # Use XGBoost

# my_model = XGBRegressor(n_estimator=1000, learning_rate=0.1)

# my_model.fit(train_X, train_y,early_stopping_rounds=5,

#              eval_set= [(val_X, val_y)], verbose=False)



# # To improve accuracy, create a new Random Forest model which you will train on all training data

# rf_model_on_full_data = RandomForestRegressor(random_state = 1)



# # fit rf_model_on_full_data on all data from the training data

# rf_model_on_full_data.fit(X, y)

# make predictions which we will submit. 

# test_preds = my_model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': pred_y})

output.to_csv('submission.csv', index=False)

# kaggle competitions submit -c home-data-for-ml-course -f submission.csv -m "Message"