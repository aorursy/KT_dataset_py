import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import lightgbm as lgb
import xgboost as xgb
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import ElasticNet
#from sklearn.neural_network import MLPRegressor
#from sklearn.linear_model import Ridge
#from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from mlxtend.regressor import StackingRegressor
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.grid_search import RandomizedSearchCV
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold
warnings.simplefilter(action='ignore', category=FutureWarning)
%matplotlib inline

print(os.listdir("../input"))
list1 = ['f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a','491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2','adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0','58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b','b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1',
        'ID','target']
list2 = ['f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a','491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2','adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0','58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b','b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1',
        'ID']
train_df = pd.read_csv("../input/train.csv",usecols = list1)
test_df = pd.read_csv("../input/test.csv",usecols = list2)
df = pd.concat([train_df, test_df], axis=0)
sub_id = test_df.ID
print("Train samples: {}, test samples: {}".format(len(train_df), len(test_df)))
del train_df, test_df
gc.collect()
X_train = df[df['target'].notnull()].drop(["ID", "target"], axis=1)
y_train = df[df['target'].notnull()].target
X_test = df[df['target'].isnull()].drop(["ID"], axis=1)
y_train = np.log1p(df[df['target'].notnull()].target)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear', 'metric': 'rmse', 'learning_rate': 0.005, 'max_depth': 7, 'subsample': 0.9, 'colsample_bytree': 0.64,'alpha':0, 'silent': True, 'random_state':42}
# {'objective': 'reg:linear','eval_metric': 'rmse','eta': 0.005,'max_depth': 15,'subsample': 0.7,'colsample_bytree': 0.5,'alpha':0,'random_state':42,'silent': True}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb
pred_test, model = run_xgb(dev_X, dev_y, val_X, val_y, X_test.drop(["target"],axis=1))
print("GB Training Completed...")
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "boosting_type":'gbdt',
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 180,
        "learning_rate" : 0.008,
        "bagging_fraction" : 0.5,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 4,
        "bagging_seed" : 42,
        "max_depth" : -1,
        "reg_alpha" : 0.3,
        "reg_lambda" : 0.1,
        "min_child_weight" : 10,
        "zero_as_missing" : True,
        "verbose" : 1,
        "random_seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 
                      num_boost_round = 10000,
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=200, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
pred_test_lgb, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")
sub_xlgb = pd.DataFrame()
sub_xlgb["target"] = (pred_test*0.5 + pred_test_lgb*0.5)
sub_xlgb = sub_xgb.set_index(sub_id)
sub_xlgb.to_csv('sub_blend_xgb+lgb.csv', encoding='utf-8-sig')
