#Import starter packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Import train and test sets

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#Combine into one since we are building random forests and gradient boosted models so we want

#to capture all levels of the features that could be present

train['test_train'] = 'train'

test['test_train'] = 'test'



train_test = pd.concat([train, test]).reset_index()
#Basic Feature Engineering

train_test['have_pool'] = train_test.apply(lambda row: 1 if row['PoolArea'] > 0 else 0, axis = 1)

train_test['have_garage'] = train_test.apply(lambda row: 0 if np.isnan(row['GarageYrBlt']) else 1, axis = 1)

train_test['built_after1999'] = train_test.apply(lambda row: 0 if row['YearBuilt'] < 2000 else 1, axis = 1)

train_test['remod_after2004'] = train_test.apply(lambda row: 0 if row['YearRemodAdd'] < 2005 else 1, axis = 1)

#Strip out the response for y_train and drop from predictors



y_train = np.log(train.SalePrice)



train_test = train_test.drop('SalePrice', axis = 1)



#For float64 type, impute values



#Put missing for NA's in strings

for i in train_test.columns:

    if(train_test[i].dtypes == 'object'):

        train_test[i] = train_test[i].fillna('missing')

              

#Put -999999999 for NA's in numeric

for i in train_test.columns:

    if(train_test[i].dtypes != 'object'):

        train_test[i] = train_test[i].fillna(-9999999999)
#Create dummy variables to feed into models

train_test_dummies = pd.get_dummies(train_test.select_dtypes(include = 'object'))
#Clean data for training and testing

train_test_numeric = train_test.select_dtypes(include = ["int64", "float64"])

X_train_test = pd.concat([train_test_numeric, train_test_dummies], axis = 1)

X_train = X_train_test[X_train_test.test_train_train == 1]

X_test = X_train_test[X_train_test.test_train_test == 1]

X_train = X_train.drop(["test_train_train", 'test_train_train'], axis = 1)

X_test = X_test.drop(["test_train_train", 'test_train_train'], axis = 1)



#Ensure shape is as expected

print(X_train.shape)

print(X_test.shape)
#Grid search to find optimal hyperparameters for xgboost model

#from sklearn.model_selection import GridSearchCV



#model_params = {'learning_rate': [0.01],

                #'n_estimators': [1000, 2000, 3000],

                #'max_depth' : [3],

                #'subsample' : [0.8]}



#grid = GridSearchCV(xgb_reg, param_grid = model_params, scoring = 'neg_mean_squared_error', cv = 10)



#grid.fit(X_train, y_train)

#Instantiate xgboost model with best hyperparameters found from grid search

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score

gb_final = XGBRegressor(learning_rate = 0.01,

                                     n_estimators = 2000,

                                     max_depth = 3,

                                     subsample = 0.8,

                                     random_state = 777)



#Cross Validate Score: Try to beat 0.0152508 

score_gb = cross_val_score(gb_final, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)

print(score_gb.mean() * -1 ** (1/2))

gb_final.fit(X_train, y_train)
#Look at how rmse varies when adding variables by feature importance for xgboost model

xgb_vars_ordered = list(gb_final.get_booster().get_score(importance_type = 'gain').keys())

#scores_xgb = []

#vars_to_use = xgb_vars_ordered[0:150]

#for i in xgb_vars_ordered[150:171]:

    #vars_to_use.append(i)

    #xgb_cross_val_score = cross_val_score(gb_final, X_train[vars_to_use], y_train, scoring = 'neg_mean_squared_error', cv = 5)

    #scores_xgb.append(xgb_cross_val_score.mean() * -1 ** (1/2))



#xgb_df_score = pd.DataFrame({'var': vars_to_use[-len(scores_xgb):], 'score':scores_xgb})



#print(xgb_df_score)
#Vars obtained from Feature Selection

vars_to_use = xgb_vars_ordered[0:161]
#Plug vars to use in model

xgb_final_final = gb_final.fit(X_train[vars_to_use], y_train)
#Visual of predictions versus actuals on training set

import matplotlib.pyplot as plt

pred_y = gb_final.predict(X_train[vars_to_use])



plt.scatter(pred_y, y_train)

plt.show()
#Prepare test scores

final_pred_y = gb_final.predict(X_test[vars_to_use])



submission = pd.DataFrame({'ID': test.Id, 'Saleprice': final_pred_y})



submission.Saleprice = np.exp(submission.Saleprice)



print(submission.head())
#Write final scores to csv

submission.to_csv('xgbsubmission.csv', index = False)