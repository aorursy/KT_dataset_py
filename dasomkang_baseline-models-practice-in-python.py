import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')  # train dataframe
test_df  = pd.read_csv('../input/test.csv')   # test dataframe
train_df.head(n=10)
#test_df.head()
# training set
print ("Training set:")
n_data  = len(train_df)
n_features = train_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))

# test set
print ("\nTest set:")
n_data  = len(test_df)
n_features = test_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))
train_df.info()
#test_df.info()
#### Check if there are any NULL values in training Data
print("Total Training Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#### Check if there are any NULL values in test Data
print("Total Test Features with NaN Values = " + str(test_df.columns[test_df.isnull().sum() != 0].size))
if (test_df.columns[test_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(test_df.columns[test_df.isnull().sum() != 0])))
    test_df[test_df.columns[test_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
#Example
##step 1) Select multiple columns
newdf = train_df[train_df.columns[25:30]]
newdf
##step 2) Check if all the values in each row/column are 0 
#newdf != 0
(newdf != 0).any(axis=0)
# column 'd5308d8bc' has only 0 values
##step 3) Double check the column'd5308d8bc' in raw data(train_df)
total = train_df['d5308d8bc'].sum()
print(total)
##step 4) Drop the columns where all values are zero 
newdf.loc[:, (newdf != 0).any(axis=0)]
# 1) training data
train_df = train_df.loc[:, (train_df != 0).any(axis=0)]
#train_df
# 256 columns are dropped

# 1) test data
test_df = test_df.loc[:, (test_df != 0).any(axis=0)]
#test_df
# 1 column is dropped
print("Train set size: {}".format(train_df.shape))
print("Test set size: {}".format(test_df.shape))
#step 1) set x and y
#a) x in train data
## axis = 1 means a row
X_train = train_df.drop(["ID", "target"], axis=1)

#b) y in train data
# #np.log1p :  log(1 + x)
Y_train = np.log1p(train_df["target"].values)
#X_train.head()
#Y_train

#c) x in test data
X_test = test_df.drop(["ID"], axis=1)

# step 2) split train_df 
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size = 0.3, random_state = 42)
# step 1) set lgb function
## meaning of expm1
#lgb function 
def run_lgb(train_x, train_y, valid_x, valid_y, test_x):
    params = {
        "boosting_type":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 40,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "random_seed": 42
    }
    
    lgtrain = lgb.Dataset(train_x, label=train_y)
    lgval = lgb.Dataset(valid_x, label=valid_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_x, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
# step 2) Training LGB
pred_test, model, evals_result = run_lgb(x_train, y_train, x_valid, y_valid, X_test)
print("LightGBM Training Completed...")
# step 3) feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])
# step 1) set XGB function 
def run_xgb(train_x, train_y, valid_x, valid_y, test_x):
    params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.005,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.5,
          'alpha':0,
          'random_state': 42, 
          'silent': True}
    
    tr_data = xgb.DMatrix(train_x, train_y)
    va_data = xgb.DMatrix(valid_x, valid_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 10, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_x)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb
# step 2) Training XGB
pred_test_xgb, model_xgb = run_xgb(x_train, y_train, x_valid, y_valid, X_test)
print("XGB Training Completed...")

# combine the predictions from two above models and submit it to Kaggle competition
sub = pd.read_csv('../input/sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test
#sub_lgb

sub_xgb = pd.DataFrame()
sub_xgb["target"] = pred_test_xgb
#sub_xgb

sub["target"] = (sub_lgb["target"] + sub_xgb["target"])/2

print(sub.head())
sub.to_csv('sub_lgb_xgb.csv', index=False)