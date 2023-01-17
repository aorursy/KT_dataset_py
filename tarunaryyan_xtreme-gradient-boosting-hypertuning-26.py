# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV

from pandas import DataFrame

from scipy import stats

from matplotlib.pylab import rcParams

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
## Reading files



train = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv")

test = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv")

# hist_txns = pd.read_csv("../input/historical_transactions.csv")

# new_txns = pd.read_csv("../input/new_merchant_transactions.csv")

# merchants = pd.read_csv("../input/elo-merchant-category-recommendation/merchants.csv")

# all_txns = pd.read_csv("../input/transactions-data/all_txns.csv")

# all_txns = pd.concat([hist_txns, new_txns], ignore_index=True)

# all_txns.to_csv("all_txns.csv", index=False)

grouped_txns = pd.read_pickle("../input/transactions-data-with-pivoted-category-columns/grouped_txns.pickle")
# txns_subset = all_txns[["card_id", "purchase_amount", "month_lag",

#                         "installments", "authorized_flag", "category_1", "category_3"]]

# # print (txns_subset.head(50))

# grouped_txns = txns_subset.groupby("card_id").agg({

#     "purchase_amount": [np.sum, np.mean],

#     "month_lag": [np.sum, np.mean],

#     "installments": [np.sum, np.mean],

#     "authorized_flag": {"Y": lambda x: x[x == 'Y'].count(),

#                         "N": lambda x: x[x == 'N'].count()},

#     "category_1": {"Y": lambda x: x[x == 'Y'].count(),

#                    "N": lambda x: x[x == 'N'].count()},

#     "category_3": {"A": lambda x: x[x == 'A'].count(),

#                    "B": lambda x: x[x == 'B'].count(),

#                    "C": lambda x: x[x == 'C'].count()}

# })

# grouped_txns.columns = ["_".join(x) for x in grouped_txns.columns.ravel()]

# grouped_txns = grouped_txns.reset_index()

# # print (grouped_txns.shape)

# print (grouped_txns.head(5))

# # print (list(grouped_txns.columns))
# grouped_txns.to_pickle("grouped_txns.pickle")
train_data = train.join(grouped_txns.set_index("card_id"), on="card_id")

test_data = test.join(grouped_txns.set_index("card_id"), on="card_id")



# print (train_data.shape)

# print (test_data.shape)
# Extracting number of active months



train_data['first_active_month'] = pd.to_datetime(train_data['first_active_month'])

test_data['first_active_month'] = pd.to_datetime(test_data['first_active_month'])

current_date = datetime.datetime.today()



def _number_of_active_months(design_matrix):

    design_matrix['number_of_active_months'] = ((current_date - design_matrix['first_active_month']) / 30.).dt.days

    return design_matrix



train_mod = _number_of_active_months(train_data)

test_mod = _number_of_active_months(test_data)



## Imputing missing values in test data

fill_na_with_mean = np.mean(test_mod['number_of_active_months'])

test_mod['number_of_active_months'].fillna(fill_na_with_mean, inplace=True)
# Normalizing continuous features



features_to_norm = ['number_of_active_months', 'month_lag_sum', 'month_lag_mean',

                    'installments_sum', 'installments_mean', 'authorized_flag_Y',

                    'authorized_flag_N', 'category_1_Y', 'category_1_N',

                    'category_3_A', 'category_3_B', 'category_3_C']

for feature in features_to_norm:

    mean_value = np.mean(train_mod[feature])

    stdev = np.std(train_mod[feature])

    train_mod[feature] = train_mod[feature].apply(lambda x: (x - mean_value) / (stdev))

    test_mod[feature] = test_mod[feature].apply(lambda x: (x - mean_value) / (stdev))
# Creating dummies for categorical features



cat_features = ['feature_1', 'feature_2', 'feature_3']

response = ['target']



def _create_dummies_for_categorical_features(design_matrix):

    """Create dummies for categorical features"""

    design_matrix = pd.get_dummies(design_matrix, prefix=cat_features,

                                   columns=cat_features)

    return design_matrix



design_matrix_train = train_mod[list(test_mod.columns) + response]

design_matrix_test = test_mod[list(test_mod.columns)]



train_with_dummies = _create_dummies_for_categorical_features(design_matrix_train)

test_with_dummies = _create_dummies_for_categorical_features(design_matrix_test)
# Defining predictors for the model



# predictors = ['purchase_amount_mean', 'category_1_Y', 'category_1_N',

#               'month_lag_mean', 'installments_mean',

#               'authorized_flag_N', 'authorized_flag_Y', 'number_of_active_months',

#               'feature_2_1', 'feature_2_2', 'feature_2_3', 'feature_3_0', 'feature_3_1',

#               'category_3_A', 'category_3_B', 'category_3_C',

#               'feature_1_1', 'feature_1_2', 'feature_1_3', 'feature_1_4', 'feature_1_5']



predictors = [x for x in test_with_dummies.columns if x not in [

    'card_id', 'first_active_month', 'feature_2_1', 'feature_2_2']]

print (len(predictors))

print (predictors)
# xgb_param = xgb1.get_xgb_params()

# xgtrain = xgb.DMatrix(train_with_dummies[predictors], label=train_with_dummies['target'])

# cv_result = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'],

#                    nfold=5, metrics='rmse', early_stopping_rounds=50, verbose_eval=True)

# xgb1.set_params(n_estimators=cv_result.shape[0])
# print (cv_result[['train-rmse-mean', 'test-rmse-mean']])
# xgb1.fit(train_with_dummies[predictors], train_with_dummies['target'], eval_metric='rmse')

# dtrain_predictions = xgb1.predict(train_with_dummies[predictors])

# print ("MSE: %.4g" % metrics.mean_squared_error(train_with_dummies['target'], dtrain_predictions))
# Model assessment using cross-validation



regressor = XGBRegressor(seed=9)

parameters = {'learning_rate': [0.01],

              'n_estimators': [600],

              'max_depth': [4],

              'min_child_weight': [1],

              'gamma': [0],

              'subsample': [0.9],

              'colsample_bytree': [0.8],

              'objective': ['reg:linear'],

              'reg_alpha': [0.01]

             }



model_grid = GridSearchCV(regressor,

                          param_grid=parameters,

                          scoring= "neg_mean_squared_error",

                          cv=10,

                          n_jobs=-1,

                          verbose=2,

                          iid=False

                         )



model_grid.fit(train_with_dummies[predictors], train_with_dummies['target'])

print ("Best: %f using %s" % (model_grid.best_score_, model_grid.best_params_))
# Cross-validation results



cv_results = model_grid.cv_results_

results = DataFrame.from_dict(cv_results, orient='columns')

results.sort_values(['mean_test_score'], ascending=False, inplace=True)

print (results[['param_n_estimators', 'mean_test_score']])
## Validation using best estimator



# best_model = model_grid.best_estimator_

# validation_data['target_predicted'] = best_model.predict(validation_data[predictors])

# validation_data.to_csv('validation_data.csv', index=False)



# RMSE

mean_squared_error = (-1.) * model_grid.best_score_

rmse = np.sqrt(mean_squared_error)

print ('RMSE: ', rmse)
# Fitting model with the best (chosen) estimator



rcParams['figure.figsize'] = 20, 8

best_model = model_grid.best_estimator_

model = best_model.fit(train_with_dummies[predictors], train_with_dummies['target'])

feat_imp = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')

# feat_imp = feat_imp[:99]

# feat_imp.drop("feature_2_2", inplace=True)

print (list(feat_imp.index))

print (feat_imp)

print (feat_imp.sum())

feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')



# Model score

# score = best_model.score(train_with_dummies[predictors], train_with_dummies['target'])

# print ()

# print ('Model score: ', score)



# Predictions on test data

test_with_dummies['target'] = model.predict(test_with_dummies[predictors])

# print (test_with_dummies.head(10))
# Submissions



submission_file = test_with_dummies[['card_id', 'target']]

# print (submission_file.head(20))



submission_file.to_csv('submission_26_' + str(rmse) + '.csv', index=False)