### Import required libraries

import numpy as np
import pandas as pd
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import clone

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
train_df.info()
test_df.info()
#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#### Check if there are any NULL values in Test Data
print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
def transformer(y, func=None):
    """Transforms target variable and prediction"""
    if func is None:
        return y
    else:
        return func(y)
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def stacking_regression(models, meta_model, X_train, y_train, X_test,
             transform_target=None, transform_pred=None,
             metric=None, n_folds=5, average_fold=True,
             shuffle=False, random_state=42, verbose=1):

    # Specify default metric for cross-validation
    if metric is None:
        metric = rmse

    # Print metric
    if verbose > 0:
        print('metric: [%s]\n' % metric.__name__)

    # Split indices to get folds
    kf = KFold(n_splits = n_folds, shuffle = shuffle, random_state = random_state)

    if X_train.__class__.__name__ == "DataFrame":
    	X_train = X_train.as_matrix()
    	X_test = X_test.as_matrix()

    # Create empty numpy arrays for stacking features
    S_train = np.zeros((X_train.shape[0], len(models)))
    S_test = np.zeros((X_test.shape[0], len(models)))

    # Loop across models
    for model_counter, model in enumerate(models):
        if verbose > 0:
            print('model %d: [%s]' % (model_counter, model.__class__.__name__))

        # Create empty numpy array, which will contain temporary predictions for test set made in each fold
        S_test_temp = np.zeros((X_test.shape[0], n_folds))
        # Loop across folds
        for fold_counter, (tr_index, te_index) in enumerate(kf.split(X_train, y_train)):
            X_tr = X_train[tr_index]
            y_tr = y_train[tr_index]
            X_te = X_train[te_index]
            y_te = y_train[te_index]
            # Clone the model because fit will mutate the model.
            instance = clone(model)
            # Fit 1-st level model
            instance.fit(X_tr, transformer(y_tr, func = transform_target))
            # Predict out-of-fold part of train set
            S_train[te_index, model_counter] = transformer(instance.predict(X_te), func = transform_pred)
            # Predict full test set
            S_test_temp[:, fold_counter] = transformer(instance.predict(X_test), func = transform_pred)

            # Delete temperatory model
            del instance

            if verbose > 1:
                print('    fold %d: [%.8f]' % (fold_counter, metric(y_te, S_train[te_index, model_counter])))

        # Compute mean or mode of predictions for test set
        if average_fold:
            S_test[:, model_counter] = np.mean(S_test_temp, axis = 1)
        else:
            model.fit(X_train, transformer(y_train, func = transform_target))
            S_test[:, model_counter] = transformer(model.predict(X_test), func = transform_pred)

        if verbose > 0:
            print('    ----')
            print('    MEAN RMSE:   [%.8f]\n' % np.sqrt((metric(y_train, S_train[:, model_counter]))))

    # Fit our second layer meta model
    meta_model.fit(S_train, transformer(y_train, func = transform_target))
    # Make our final prediction
    stacking_prediction = transformer(meta_model.predict(S_test), func = transform_pred)

    return stacking_prediction
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)
elastic_net = ElasticNet(alpha = 0.02, l1_ratio = 0.15, random_state = 42)
elastic_net.fit(X_train,y_train)
rf_tree = RandomForestRegressor(n_estimators = 1000,
                                max_features = "sqrt",
                                max_depth = 15,
                                min_samples_split = 20,
                                min_samples_leaf = 5,
                                bootstrap = True,
                                random_state = 42)
rf_tree.fit(X_train, y_train)
gb_tree = GradientBoostingRegressor(max_depth = 5, 
                                    learning_rate = 0.01, 
                                    n_estimators = 1000,
                                    min_samples_split = 15,
                                    max_features = "sqrt",
                                    min_samples_leaf = 3,
                                    random_state=42)
gb_tree.fit(X_train, y_train)
xgb_tree = XGBRegressor(max_depth = 10, 
                        learning_rate = 0.01, 
                        n_estimators = 1000,
                        min_child_weight = 5,
                        reg_alpha = 0.03, 
                        random_state=42)
xgb_tree.fit(X_train, y_train)
lgb_tree = lgb.LGBMRegressor(learning_rate = 0.01, 
                             num_leaves = 40,
                             n_estimators = 1000,
                             bagging_fraction = 0.6,
                             feature_fraction = 0.5,
                             random_state=42)
lgb_tree.fit(X_train, y_train)
models = [rf_tree, gb_tree, xgb_tree, lgb_tree]
meta_model = elastic_net
y_predicted = stacking_regression(models, meta_model, X_train, y_train, X_test,
             metric=None, n_folds=5, average_fold=True,
             shuffle=True, random_state=42, verbose=2)
y_pred = np.expm1(y_predicted)
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = y_pred

print(sub.head())
sub.to_csv('sub_stacking.csv', index=False)