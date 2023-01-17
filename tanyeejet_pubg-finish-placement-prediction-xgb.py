import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")

df_train = df_train.dropna(axis=0)
df_test = df_test.dropna(axis=0)

features = df_train.columns.drop(["winPlacePerc", "Id", "groupId", "matchId"])
train_X = df_train[features]
train_y = df_train['winPlacePerc']
test_X = df_test[features]

#one hot encode
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)
from xgboost import XGBRegressor
# learningrate 0.01, maxdepth 6, colsamplebytree 1, nestimators 1000
XGBRegressor_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.01, max_delta_step=0,
       max_depth=6, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=42,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=0, subsample=1 )
XGBRegressor_model.fit(train_X, train_y, verbose=False)
predict_y = XGBRegressor_model.predict(test_X)
output = pd.DataFrame({'Id': df_test.Id,
                       'winPlacePerc': predict_y})

output.to_csv('submission.csv', index=False)
output