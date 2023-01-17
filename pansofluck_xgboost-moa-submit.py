!pip install scikit-learn

!pip install xgboost
import xgboost as xgb

import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
def preprocess(df):

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)



del train_targets_scored['sig_id']
X, y = train, train_targets_scored
X
y
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
X_train.shape, y_train.shape
models_list = []

res = y_test.copy()



for idx, column_name in enumerate(y_train.columns):

    print(idx, end='\r')

    xg_reg = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(X_train,y_train[column_name])

    res[column_name] = xg_reg.predict(X_test)

    ss[column_name] = xg_reg.predict(test)



#     models_list.append(xg_reg)

    del xg_reg
res.describe()
def rmse(y_pred, y_true):

    return np.sqrt(np.mean(np.power(y_true-y_pred, 2)))

def logloss(y_pred, y_true):

    return -np.mean(y_true.dot(np.log(y_pred).T) + (1-y_true).dot(np.log(1-y_pred).T))

print("RMSE: %f" % (rmse(res.values, y_test.values)))

print("LOGLOSS: %f" % (logloss(res.values, y_test.values)))
ss.to_csv('submission.csv', index=False)