# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# read the data and store in DataFrame

train_data = pd.read_csv('../input/train.csv')
# Extract the target variable

y = train_data.SalePrice

train_data.drop('SalePrice', axis=1, inplace=True)
# The final competition score is based on the MSE between the log of the test predictions and the log of the true SalePrice.

# With that in mind, always train to fit logy.

logy = np.log(y)

# Predictions will need to be of y, however, so for the final test submission, take the exponent of its output.
# Separate the Id column from the predictive features

X = train_data.drop('Id', axis=1)
# Compute a hash of each instance's identifier,

# keep only the last byte of the hash,

# and put the instance in the val set if the value of that byte is < val_ratio*256.

# (Hands-On Machine Learning with Scikit-Learn & TensorFlow, pg. 50 of my copy)



VAL_RATIO = 0.2

import hashlib

val_set_mask = train_data.Id.apply(lambda id : hashlib.md5(np.int64(id)).digest()[-1] < VAL_RATIO * 256)
# Separate a val set

val_X = X.loc[val_set_mask]

train_X = X.loc[~val_set_mask]

val_y = y.loc[val_set_mask]

train_y = y.loc[~val_set_mask]

val_logy = logy.loc[val_set_mask]

train_logy = logy.loc[~val_set_mask]
X_numeric = X.select_dtypes(include=[np.number]).drop('MSSubClass', axis=1)

X_categorical = X.drop(X_numeric.columns, axis=1)



num_cols = X_numeric.columns

cat_cols = X_categorical.columns
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import RobustScaler, OneHotEncoder

from sklearn.impute import SimpleImputer



num_pipeline = Pipeline([

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])



cat_pipeline = Pipeline([

    ('cat_nan_filler', SimpleImputer(strategy='constant', fill_value='not_in_data')),

    ('cat_onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor_pipeline = ColumnTransformer([

    ('num_pipeline', num_pipeline, num_cols),

    ('cat_pipeline', cat_pipeline, cat_cols)

])
X = preprocessor_pipeline.fit_transform(X)

train_X = preprocessor_pipeline.transform(train_X)

val_X = preprocessor_pipeline.transform(val_X)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor



N_ESTIMATORS = 100

rf_regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, criterion='mse')

et_regressor = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, criterion='mse')
rf_regressor.fit(train_X, train_logy)
from sklearn.metrics import mean_squared_error



rf_val_mse = mean_squared_error(rf_regressor.predict(val_X), val_logy)

rf_val_mse
et_regressor.fit(train_X, train_logy)
et_val_mse = mean_squared_error(et_regressor.predict(val_X), val_logy)

et_val_mse
from sklearn.model_selection import GridSearchCV
rf_paramgrid = {'min_samples_leaf': [1,2,3,4,5],}

et_paramgrid = {'min_samples_leaf': [1,2,3,4,5],}



rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
def gridsearch_fit_and_print_results(gridsearch, data, target):

    gridsearch.fit(data, target)

    

    print("Best parameters set found on development set:")

    print()

    print(gridsearch.best_params_)

    print()

    print("Grid scores on development set:")

    print()

    means = gridsearch.cv_results_['mean_test_score']

    stds = gridsearch.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gridsearch.cv_results_['params']):

        print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    print()
gridsearch_fit_and_print_results(rf_gridsearch, X, logy)
gridsearch_fit_and_print_results(et_gridsearch, X, logy)
rf_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4, 5]}

et_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4, 5]}



rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)

et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)
gridsearch_fit_and_print_results(rf_gridsearch, X, logy)
gridsearch_fit_and_print_results(et_gridsearch, X, logy)
rf_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4], 'max_features': ['sqrt']}

et_paramgrid = {'min_samples_leaf': [1,2], 'max_depth': [None, 10, 100], 'min_samples_split': [2, 3, 4], 'max_features': ['sqrt']}



rf_gridsearch = GridSearchCV(rf_regressor, rf_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)

et_gridsearch = GridSearchCV(et_regressor, et_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False, verbose=1, n_jobs=-1)
gridsearch_fit_and_print_results(rf_gridsearch, X, logy)
gridsearch_fit_and_print_results(et_gridsearch, X, logy)
# Best models so far:

best_model_et = ExtraTreesRegressor(n_estimators=100, criterion='mse', min_samples_leaf=2)

best_model_rf = RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=3)

# Mean validation score: -0.02010 (+/-0.00867) (changes randomly by a small amount from run to run)
# Ridge:

# " Minimizes the objective function:

# ||y - Xw||^2_2 + alpha * ||w||^2_2 "



# Lasso:

# " The optimization objective for Lasso is:

# (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1 "



# ElasticNet:

# "Minimizes the objective function:

# 1 / (2 * n_samples) * ||y - Xw||^2_2

# + alpha * l1_ratio * ||w||_1

# + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

# 

# l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable, unless you supply your own sequence of alpha."



from sklearn.linear_model import Ridge, Lasso, ElasticNet



ridge_regressor = Ridge(alpha=1.0, max_iter=10000) # default alpha value, 10x max_iter default (for sag solver)

lasso_regressor = Lasso(alpha=1.0, max_iter=10000) # default alpha value, 10x max_iter default (for sag solver)

en_regressor = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000) # default alpha and l1_ratio values, 10x max_iter default (for sag solver)
ridge_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0],}

lasso_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0],}

en_paramgrid = {'alpha': [1.0, 0.3, 0.1, 3.0, 10.0], 'l1_ratio': [0.5, 0.75, 0.25, 0.9, 0.1]}
ridge_gridsearch = GridSearchCV(ridge_regressor, ridge_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

en_gridsearch = GridSearchCV(en_regressor, en_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)
gridsearch_fit_and_print_results(ridge_gridsearch, X, logy)
gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)
lasso_paramgrid = {'alpha': [0.1, 0.03, 0.01, 0.003, 0.001],}

lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)
lasso_paramgrid = {'alpha': [0.001, 0.0003, 0.0001, 0.00003, 0.00001],}

lasso_gridsearch = GridSearchCV(lasso_regressor, lasso_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

gridsearch_fit_and_print_results(lasso_gridsearch, X, logy)
en_paramgrid = {'alpha': [0.0003, 0.001, 0.003], 'l1_ratio': [0.5, 0.75, 0.25, 0.9, 0.1]}

en_gridsearch = GridSearchCV(en_regressor, en_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

gridsearch_fit_and_print_results(en_gridsearch, X, logy)
from sklearn.linear_model import LinearRegression



lin_regressor = LinearRegression()

lin_paramgrid = {'fit_intercept': [True, False]}

lin_gridsearch = GridSearchCV(lin_regressor, lin_paramgrid, scoring='neg_mean_squared_error', cv=5, return_train_score=False)

gridsearch_fit_and_print_results(lin_gridsearch, X, logy)
best_models = [ElasticNet(alpha=0.001, l1_ratio=0.5),

               ExtraTreesRegressor(n_estimators=100, criterion='mse', min_samples_leaf=2),

              RandomForestRegressor(n_estimators=100, criterion='mse', min_samples_split=3)]