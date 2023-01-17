import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
%%time
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print(df_train.shape)
print(df_test.shape)
target = 'winPlacePerc'
train_columns = list(df_train.columns)
train_columns.remove(target)

X = df_train[train_columns]
y = df_train[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
dval= xgb.DMatrix(X_val.values, label=y_val.values)

params = {
    "max_depth" : 5,
    'eval_metric': ['mae'],
}

clf = xgb.train(params, dtrain, evals=[(dtrain, "train"),(dval, 'val')], num_boost_round = 50)
dtest = xgb.DMatrix(df_test.values)
df_test['winPlacePerc'] = clf.predict(dtest)
submission = df_test[['Id', 'winPlacePerc']]
submission.to_csv('submission.csv', index=False)
