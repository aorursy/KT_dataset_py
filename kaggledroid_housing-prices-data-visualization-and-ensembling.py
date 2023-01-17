import pandas as pd

import numpy as np

from math import sqrt

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error

import category_encoders as ce

from sklearn.impute import KNNImputer

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

import operator
#Loading the training data

X_path = '../input/house-prices-advanced-regression-techniques/train.csv'

X = pd.read_csv(X_path, index_col='Id')

test_path = "../input/house-prices-advanced-regression-techniques/test.csv"

X_test = pd.read_csv(test_path, index_col='Id')



#Dropping rows with missing value for target

X.dropna(axis=0, subset=['SalePrice'], inplace=True)



#Separating target from features

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)



X.head()
X.shape
X.columns
#Identifying columns with numerical data

num_cols = [col for col in X.columns if X[col].dtypes in ['int64', 'float64']]

print(num_cols)

print()

print(len(num_cols))
#Identifying columns with categorical data

cat_cols = [col for col in X.columns if X[col].dtypes=='object']

print(cat_cols)

print()

print(len(cat_cols))
#Identifying columns with missing values

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

print(cols_with_missing)

print()

print(len(cols_with_missing))



cols_with_missing_test = [col for col in X_test.columns if X_test[col].isnull().any()]

print(cols_with_missing_test)

print()

print(len(cols_with_missing_test))
#Identifying the number of missing values for each column in cols_with_missing

for col in cols_with_missing:

    print(col + " : " + str(X[col].isnull().sum()))
#We have a total of 1460 rows.

#And Alley, PoolQC, Fence and MiscFeature have more than a thousand missing values.

#Thus it is reasonable to drop these columns from the dataset.



X.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

X_test.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
#Updating the cols_with_missing list and cat_cols list

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

cat_cols = [col for col in X.columns if X[col].dtypes=='object']



cols_with_missing_test = [col for col in X_test.columns if X_test[col].isnull().any()]

cat_cols_test = [col for col in X_test.columns if X_test[col].dtypes=='object']
for col in cat_cols:

    plt.figure(figsize=(8,6)) 

    sns.barplot(x=X[col], y=y) 
#From the above plots we can see that the different values in LotConfig, LandSlope have almost the same SalePrice.

#Thus we can now drop these columns from the dataset as they wouldn't affect the predictions in any considerable way.

X.drop(['LotConfig', 'LandSlope'], axis=1, inplace=True)

X_test.drop(['LotConfig', 'LandSlope'], axis=1, inplace=True)



#Updating cat_cols

cat_cols = [col for col in X.columns if X[col].dtypes=='object']

cat_cols_test = [col for col in X_test.columns if X_test[col].dtypes=='object']
for col in num_cols:

    plt.figure()

    sns.regplot(x=X[col], y=y, line_kws={'color':'#f35588'})
#From the above plots we can see that YrSold, MoSold, MiscVal, 3SsnPorch, BsmtHalfBath, LowQualFinSF, 

#BsmtFinS2 and MSSubClass have a weak correlation with the target. Thus it is safe to drop these columns as well.

X.drop(['YrSold', 'MoSold', 'MiscVal', '3SsnPorch', 'BsmtHalfBath', 'LowQualFinSF', 'BsmtFinSF2', 'MSSubClass'], 

       axis=1, inplace=True)

X_test.drop(['YrSold', 'MoSold', 'MiscVal', '3SsnPorch', 'BsmtHalfBath', 'LowQualFinSF', 'BsmtFinSF2', 'MSSubClass'], 

       axis=1, inplace=True)



#Updating num_cols

num_cols = [col for col in X.columns if X[col].dtypes in ['int64', 'float64']]

num_cols_test = [col for col in X_test.columns if X_test[col].dtypes in ['int64', 'float64']]
X.shape
#Before we proceed with any further preprocessing, we will first split the data into training and validation data.

#This will prevent any data leakage.



X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
pd.options.mode.chained_assignment = None  # default='warn'
#Identifying categorical columns that have missing values

cat_cols_with_missing = [col for col in cat_cols if col in cols_with_missing]

cat_cols_with_missing_test = [col for col in cat_cols_test if col in cols_with_missing_test]



cat_imputer = SimpleImputer(strategy='most_frequent')



#Creating DataFrame with imputed values for categorical columns

X_train_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols_with_missing]))

X_valid_cat_imputed = pd.DataFrame(cat_imputer.transform(X_valid[cat_cols_with_missing]))



#Putting the labels back

X_train_cat_imputed.columns = X_train[cat_cols_with_missing].columns

X_valid_cat_imputed.columns = X_valid[cat_cols_with_missing].columns



#Adding the index

X_train_cat_imputed.index = X_train.index

X_valid_cat_imputed.index = X_valid.index



#Dropping the categorical columns with missing values from initial training data

X_train.drop(cat_cols_with_missing, axis=1, inplace=True)

X_valid.drop(cat_cols_with_missing, axis=1, inplace=True)



#Creating a new DataFrame that does not contain any missing values for any categorical column

X_train_no_missing_cat = pd.concat([X_train, X_train_cat_imputed], axis=1)

X_valid_no_missing_cat = pd.concat([X_valid, X_valid_cat_imputed], axis=1)



#Defining imputer for test data and imputing the values

cat_imputer_for_test = SimpleImputer(strategy='most_frequent')



cat_imputer_for_test.fit(X_train_no_missing_cat[cat_cols_with_missing_test])

X_test_cat_imputed = pd.DataFrame(cat_imputer_for_test.transform(X_test[cat_cols_with_missing_test]))

X_test_cat_imputed.columns = X_test[cat_cols_with_missing_test].columns

X_test_cat_imputed.index = X_test.index

X_test.drop(cat_cols_with_missing_test, axis=1, inplace=True)

X_test_no_missing_cat = pd.concat([X_test, X_test_cat_imputed], axis=1)
#Using CatBoostEncoder to encode categorical data

cb_enc = ce.CatBoostEncoder(cols=cat_cols)



cb_enc.fit(X_train_no_missing_cat[cat_cols], y_train)



train_encoded_cb = X_train_no_missing_cat.join(cb_enc.transform(X_train_no_missing_cat[cat_cols]).add_suffix('_cb'))

valid_encoded_cb = X_valid_no_missing_cat.join(cb_enc.transform(X_valid_no_missing_cat[cat_cols]).add_suffix('_cb'))

test_encoded_cb = X_test_no_missing_cat.join(cb_enc.transform(X_test_no_missing_cat[cat_cols]).add_suffix('_cb'))



train_encoded_cb.drop(cat_cols, axis=1, inplace=True)

valid_encoded_cb.drop(cat_cols, axis=1, inplace=True)

test_encoded_cb.drop(cat_cols, axis=1, inplace=True)
#Now we will impute the missing values in the numerical columns using KNNImputer.

imputer = KNNImputer(n_neighbors=100, weights='distance', metric='nan_euclidean')



imputed_train_num_cols = pd.DataFrame(imputer.fit_transform(train_encoded_cb[num_cols]))

imputed_valid_num_cols = pd.DataFrame(imputer.transform(valid_encoded_cb[num_cols]))

imputed_test_num_cols = pd.DataFrame(imputer.transform(test_encoded_cb[num_cols]))



imputed_train_num_cols.columns = train_encoded_cb[num_cols].columns

imputed_valid_num_cols.columns = valid_encoded_cb[num_cols].columns

imputed_test_num_cols.columns = test_encoded_cb[num_cols].columns



imputed_train_num_cols.index = train_encoded_cb[num_cols].index

imputed_valid_num_cols.index = valid_encoded_cb[num_cols].index

imputed_test_num_cols.index = test_encoded_cb[num_cols].index



train_encoded_cb.drop(num_cols, axis=1, inplace=True)

valid_encoded_cb.drop(num_cols, axis=1, inplace=True)

test_encoded_cb.drop(num_cols, axis=1, inplace=True)



train_data = train_encoded_cb.join(imputed_train_num_cols)

valid_data = valid_encoded_cb.join(imputed_valid_num_cols)

test_data = test_encoded_cb.join(imputed_test_num_cols)
ada_boost = AdaBoostRegressor(base_estimator=XGBRegressor(verbosity=0))

ada_boost.fit(train_data, y_train)
xgb_model = XGBRegressor()

xgb_model.fit(train_data, y_train)
lgb_model = lgb.LGBMRegressor()

lgb_model.fit(train_data, y_train)
rf_model = RandomForestRegressor()

rf_model.fit(train_data, y_train)
gb_model = GradientBoostingRegressor()

gb_model.fit(train_data, y_train)
et_model = ExtraTreesRegressor()

et_model.fit(train_data, y_train)
#Score of the models without any hyperparamter tuning

print("AdaBoost Score         : ",ada_boost.score(valid_data, y_valid))

print("XGBoost Score          : ",xgb_model.score(valid_data, y_valid))

print("LightGBM Score         : ",lgb_model.score(valid_data, y_valid))

print("Random Forests Score   : ",rf_model.score(valid_data, y_valid))

print("Gradient Boosting Score: ",gb_model.score(valid_data, y_valid))

print("Extra Trees Score      : ",et_model.score(valid_data, y_valid))
params_for_ada_boost = {

    "base_estimator__n_estimators": [100, 300, 500],

    "base_estimator__reg_lambda": [1, 2, 4],

    "n_estimators": [10, 30, 50],

    "learning_rate": [0.05, 0.1],   

}
random_search_ada_boost = RandomizedSearchCV(

    estimator=ada_boost,

    param_distributions=params_for_ada_boost,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_ada_boost.fit(train_data, y_train)
-1*random_search_ada_boost.best_score_
random_search_ada_boost.best_estimator_
ada_boost_tuned = AdaBoostRegressor(base_estimator=XGBRegressor(base_score=0.5, booster='gbtree',

                                              colsample_bylevel=1,

                                              colsample_bynode=1,

                                              colsample_bytree=1, gamma=0,

                                              importance_type='gain',

                                              learning_rate=0.1,

                                              max_delta_step=0, max_depth=3,

                                              min_child_weight=1, missing=None,

                                              n_estimators=500, n_jobs=1,

                                              nthread=None,

                                              objective='reg:linear',

                                              random_state=0, reg_alpha=0,

                                              reg_lambda=2, scale_pos_weight=1,

                                              seed=None, silent=None,

                                              subsample=1, verbosity=0),

                  learning_rate=0.1, loss='linear', n_estimators=10,

                  random_state=None)

ada_boost_tuned.fit(train_data, y_train)
ada_boost_tuned.score(valid_data, y_valid)
params_for_xgb_model = {

    'n_estimators': [500, 1000],

    'learning_rate': [0.03, 0.05, 0.1],

    'max_depth': [3, 6, 9],

    'gamma': [1,5]

}
random_search_xgb_model = RandomizedSearchCV(

    estimator=xgb_model,

    param_distributions=params_for_xgb_model,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_xgb_model.fit(train_data, y_train)
-1*random_search_xgb_model.best_score_
random_search_xgb_model.best_estimator_
xgb_model_tuned = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=1, gamma=1,

             importance_type='gain', learning_rate=0.03, max_delta_step=0,

             max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,

             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)



xgb_model_tuned.fit(train_data, y_train)
xgb_model_tuned.score(valid_data, y_valid)
params_for_lgb_model = {

    'n_estimators': [600, 800, 1000],

    'learning_rate': [0.03, 0.05],

}
random_search_lgb_model = RandomizedSearchCV(

    estimator=lgb_model,

    param_distributions=params_for_lgb_model,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_lgb_model.fit(train_data, y_train)
-1*random_search_lgb_model.best_score_
random_search_lgb_model.best_estimator_
lgb_model_tuned = lgb.LGBMRegressor(boosting='gbdt', boosting_type='gbdt', class_weight=None,

              colsample_bytree=1.0, importance_type='split', learning_rate=0.05,

              max_depth=-1, min_child_samples=20, min_child_weight=0.001,

              min_split_gain=0.0, n_estimators=1000, n_jobs=-1, num_leaves=31,

              objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,

              silent=True, subsample=1.0, subsample_for_bin=200000,

              subsample_freq=0)



lgb_model_tuned.fit(valid_data, y_valid)
lgb_model_tuned.score(valid_data, y_valid)
params_for_rf_model = {

    'n_estimators': [500, 1000, 1500, 2000]

}
random_search_rf_model = RandomizedSearchCV(

    estimator=rf_model,

    param_distributions=params_for_rf_model,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_rf_model.fit(train_data, y_train)
-1*random_search_rf_model.best_score_
random_search_rf_model.best_estimator_
rf_model_tuned = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',

                      max_depth=None, max_features='auto', max_leaf_nodes=None,

                      max_samples=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=2, min_weight_fraction_leaf=0.0,

                      n_estimators=2000, n_jobs=None, oob_score=False,

                      random_state=None, verbose=0, warm_start=False)



rf_model_tuned.fit(train_data, y_train)
rf_model_tuned.score(valid_data, y_valid)
params_for_gb_model = {

    'loss': ['huber'],

    'n_estimators': [500, 1000],

    'learning_rate': [0.001, 0.005]

}
random_search_gb_model = RandomizedSearchCV(

    estimator=gb_model,

    param_distributions=params_for_gb_model,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_gb_model.fit(train_data, y_train)
-1*random_search_gb_model.best_score_
random_search_gb_model.best_estimator_
gb_model_tuned = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',

                          init=None, learning_rate=0.005, loss='huber',

                          max_depth=3, max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=2,

                          min_weight_fraction_leaf=0.0, n_estimators=1000,

                          n_iter_no_change=None, presort='deprecated',

                          random_state=None, subsample=1.0, tol=0.0001,

                          validation_fraction=0.1, verbose=0, warm_start=False)



gb_model_tuned.fit(train_data, y_train)
gb_model_tuned.score(valid_data, y_valid)
params_for_et_model = {

    'n_estimators': [100, 500, 1000],

    'max_depth': [8, 16, 32, 64]

}
random_search_et_model = RandomizedSearchCV(

    estimator=et_model,

    param_distributions=params_for_et_model,

    scoring='neg_mean_absolute_error',

    n_iter=5,

    n_jobs=-1,

    cv=3,

    verbose=True

)
random_search_et_model.fit(train_data, y_train)
-1*random_search_et_model.best_score_
random_search_et_model.best_estimator_
et_model_tuned = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.0, criterion='mse',

                    max_depth=32, max_features='auto', max_leaf_nodes=None,

                    max_samples=None, min_impurity_decrease=0.0,

                    min_impurity_split=None, min_samples_leaf=1,

                    min_samples_split=2, min_weight_fraction_leaf=0.0,

                    n_estimators=500, n_jobs=None, oob_score=False,

                    random_state=None, verbose=0, warm_start=False)



et_model_tuned.fit(train_data, y_train)
et_model_tuned.score(valid_data, y_valid)
#Plotting a graph to see which model performs the best

model_score = [('AdaBoost', 0.882397846544142), ('XGBoost', 0.8986748676918312), ('LightGBM', 0.9957855010033861), 

               ('Random Forests', 0.8623363506281407), ('GradientBoosting', 0.8399577921912063), ('ExtraTrees', 0.8785945278599719)]



model_score.sort(key=operator.itemgetter(1), reverse=True)



X_data = [i[0] for i in model_score]

Y_data = [i[1] for i in model_score]



plt.figure(figsize=(10,6))

sns.barplot(x=X_data, y=Y_data)
#We will take the weighted average of the predictions made by the top three models using the function defined below.

def averagePreds(validation_data):

    preds_et = et_model_tuned.predict(validation_data)

    preds_xgb = xgb_model_tuned.predict(validation_data)

    preds_ada = ada_boost_tuned.predict(validation_data)

    

    final_preds = (preds_ada*0.6 + preds_xgb*0.2 + preds_et*0.2)

    

    return final_preds
valid_preds = averagePreds(valid_data)

print("MAE of the Ensemble: ",mean_absolute_error(valid_preds, y_valid))
print("AdaBoost MAE         : ",mean_absolute_error(ada_boost_tuned.predict(valid_data), y_valid))

print("XGBoost MAE          : ",mean_absolute_error(xgb_model_tuned.predict(valid_data), y_valid))

print("LightGBM MAE         : ",mean_absolute_error(lgb_model_tuned.predict(valid_data), y_valid))

print("Random Forests MAE   : ",mean_absolute_error(rf_model_tuned.predict(valid_data), y_valid))

print("Gradient Boosting MAE: ",mean_absolute_error(gb_model_tuned.predict(valid_data), y_valid))

print("Extra Trees MAE      : ",mean_absolute_error(et_model_tuned.predict(valid_data), y_valid))
test_preds = averagePreds(test_data)



output = pd.DataFrame({'Id':test_data.index, 'SalePrice':test_preds})



output.to_csv('submission.csv', index=False)