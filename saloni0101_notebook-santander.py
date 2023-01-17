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
train_csv=pd.read_csv("../input/train.csv")
test_csv=pd.read_csv("../input/test.csv")
train_csv=train_csv.drop(['ID'],axis=1)
test_id=test_csv['ID']
test_csv.drop(['ID'],axis=1,inplace=True)
target=train_csv['target']
train_csv.drop(['target'],axis=1,inplace=True)
target_log=np.log10(target)
target_vals=np.log1p(target_log)
print(train_csv.shape,test_csv.shape)
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
forest.fit(train_csv,target_vals)


importances = forest.feature_importances_
# make importance relative to the max importance
feature_importance = 100.0 * (importances / importances.max())
sorted_idx = np.argsort(feature_importance)
feature_names = list(train_csv.columns.values)
feature_names_sort = [feature_names[indice] for indice in sorted_idx]
pos = np.arange(sorted_idx.shape[0]) + .5
print('Top 40 features are: ')
cols=[]
for feature in feature_names_sort[::-1][:40]:
    cols.append(feature)
print(sorted(cols))
train_data=train_csv[cols]
test_data=test_csv[cols]
for df in [train_data,test_data]:
    df["sum"]=df.sum(axis=1)
    df["median"]=df.median(axis=1)
    df["mean"]=df.mean(axis=1)
    df["std"]=df.std(axis=1)
    df["max"]=df.max(axis=1)
    df["min"]=df.min(axis=1)
    df["kurtosis"]=df.kurtosis(axis=1)
print(train_data.shape,test_data.shape)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
y_target=np.log1p(target)
NUM_FOLDS = 5 #need tuned
def rmsle_cv(model):
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train_data.values)
    rmse= np.sqrt(-cross_val_score(model, train_data, y_target, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train_data, y_target, test_size=0.15, random_state=42)
params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 50,
        "max_depth" : 6,
        "learning_rate" : 0.0005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "lambda" : 0.01,
        "random_seed": 42
    }
lgtrain = lgb.Dataset(x_train, label=y_train)
lgval = lgb.Dataset(x_val, label=y_val)
evals_result = {}
model = lgb.train(params, lgtrain, 200000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=1500, 
                      verbose_eval=50, 
                      evals_result=evals_result)
pred_test_y = np.expm1(model.predict(test_data, num_iteration=model.best_iteration))
params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.0005,
          'max_depth':6, 
          'subsample': 0.7, 
          'colsample_bytree': 0.5,
          'alpha':0,
          'random_state': 42, 
          'silent': True}
    
tr_data = xgb.DMatrix(x_train,y_train)
va_data = xgb.DMatrix(x_val,y_val)
    
watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
model_xgb = xgb.train(params, tr_data, 200000, watchlist, maximize=False, early_stopping_rounds = 3000, verbose_eval=100)
dtest = xgb.DMatrix(test_data)
xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
sub = pd.DataFrame()
sub['ID'] = test_id
sub['target'] = pred_test_y
sub.to_csv('submission_lgb.csv',index=False)
sub = pd.DataFrame()
sub['ID'] = test_id
sub['target'] = xgb_pred_y
sub.to_csv('submission_xgb.csv',index=False)
sub = pd.DataFrame()
sub['ID'] = test_id
sub['target'] = np.mean([pred_test_y,xgb_pred_y])
sub.to_csv('submission_comb_xlgb.csv',index=False)




