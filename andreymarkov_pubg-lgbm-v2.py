import pandas as pd
import numpy as np
from scipy.stats import skew


train = pd.read_csv('../input/train_V2.csv')
y = train.winPlacePerc
test = pd.read_csv('../input/test_V2.csv')
del train['matchType']
del test['matchType']

all_data = pd.concat((train.loc[:,'assists':'winPoints'],
                      test.loc[:,'assists':'winPoints']))
train = train.loc[:,'assists':'winPlacePerc']


#all_data = pd.get_dummies(all_data)
#train = pd.get_dummies(train)
all_data = all_data.fillna(all_data.mean())
train = train.fillna(train.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import lightgbm as lgb
x_train,  x_valid, y_train, y_valid = train_test_split(X_train, train['winPlacePerc'],test_size=0.15, random_state=27)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=2000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                eval_metric= 'mae', verbose=1, early_stopping_rounds=50)
predictions_ = model_lgb.predict(X_test)
my_submission = pd.DataFrame({'Id':test.Id, 'winPlacePerc': predictions_})
print(predictions_[0:10])
my_submission.to_csv('submission.csv', index=False)

