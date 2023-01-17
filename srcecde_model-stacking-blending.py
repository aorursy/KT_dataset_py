# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from catboost import CatBoostRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
np.random.seed(7777)
train_df = pd.read_csv("../input/train_V2.csv")
test_df = pd.read_csv("../input/test_V2.csv")
matchTyp = ['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo',
       'normal-squad-fpp', 'crashfpp', 'flaretpp', 'normal-solo-fpp',
       'flarefpp', 'normal-duo-fpp', 'normal-duo', 'normal-squad',
       'crashtpp', 'normal-solo']
mapping = {}
for i, j in enumerate(matchTyp):
    mapping[i] = j
train_df["matchTypeMap"] = train_df["matchType"].apply(lambda x: ''.join(str(i) for i, j in mapping.items() if x == j)).map(np.int64)
test_df["matchTypeMap"] = test_df["matchType"].apply(lambda x: ''.join(str(i) for i, j in mapping.items() if x == j)).map(np.int64)
train_df.drop(["matchType"], axis =1, inplace=True)
test_df.drop(["matchType"], axis =1, inplace=True)
train_df.dropna(inplace = True)
train_df.isnull().any().any()
X = train_df.drop(["Id", "groupId", "matchId", "winPlacePerc"], axis = 1)
y = train_df["winPlacePerc"]
test = test_df.drop(["Id", "groupId", "matchId"], axis = 1)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(X, y, test_size = 0.5)
del X
del y
params = {
    'learning_rate': 0.3, 
#     'max_depth': 3,
    'num_leaves': 20,
    'feature_fraction': 0.9,
    'min_data_in_leaf': 100,
    'lambda_l2': 4,
    'objective': 'regression_l2', 
    'metric': 'mae',
    'seed': 123}
lgb_dataset = lightgbm.Dataset(X_train_s, y_train_s)
lgb_valid = lightgbm.Dataset(X_val_s, y_val_s)
lgb_model = lightgbm.train(params, lgb_dataset, num_boost_round=10000, valid_sets = lgb_valid, early_stopping_rounds=30, verbose_eval=100)
pred1 = lgb_model.predict(X_val_s)
kfold = 15
skf = KFold(n_splits=kfold, random_state=42)
pred2 = pd.DataFrame()
pred2['winPlacePerc'] = np.zeros(len(X_val_s))
cat_model = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')
cat_model.fit(X_train_s, y_train_s)
pred3 = cat_model.predict(X_val_s)
xgb_model = xgboost.XGBRegressor(max_depth=11)

for i, (train_index, test_index) in enumerate(skf.split(X_train_s, y_train_s)):
    X_train, X_valid = X_train_s.iloc[train_index], X_train_s.iloc[test_index]
    y_train, y_valid = y_train_s.iloc[train_index], y_train_s.iloc[test_index]
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae', early_stopping_rounds=100)
    vtest = xgb_model.predict(X_val_s)
    pred2['winPlacePerc'] += vtest/kfold

test_pred_lgm = lgb_model.predict(test)
test_pred_xgb = xgb_model.predict(test)
test_pred_cat = cat_model.predict(test)
stack_valid = np.column_stack((pred1, pred2['winPlacePerc'], pred3))
test_pred = np.column_stack((test_pred_lgm, test_pred_xgb, test_pred_cat))
stack_model = LinearRegression()
stack_model.fit(stack_valid, y_val_s)
test_stack_model = stack_model.predict(test_pred)
test_id = test_df["Id"].map(str)
submit_xg = pd.DataFrame({'Id': test_id, "winPlacePerc": test_stack_model} , columns=['Id', 'winPlacePerc'])
submit_xg.to_csv("submission.csv", index = False)