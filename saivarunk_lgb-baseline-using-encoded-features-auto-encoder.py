import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import time
import math

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

print(os.listdir("../input"))
train = pd.read_csv('../input/dimensionality-reduction-using-keras-auto-encoder/train_encoded.csv')
test = pd.read_csv('../input/dimensionality-reduction-using-keras-auto-encoder/test_encoded.csv')
def rms(y_actual, y_predicted):
    return math.sqrt(mean_squared_error(y_actual, y_predicted))
print("Train shape: ", train.shape)
print("Test shape: ", test.shape)
train.isnull().sum(axis=1).sum()
Y = np.log(train.target+1)

train.drop(['target'], axis=1, inplace=True)
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 10,
    'num_leaves': 32,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'learning_rate': 0.001,
    'verbose': 1
}
Y_target = []
scores = []

for fold_id,(train_idx, val_idx) in enumerate(KFold(n=train.shape[0], n_folds=10, random_state=1)):
    print('FOLD:',fold_id)
    X_train = train.values[train_idx]
    y_train = Y.values[train_idx]
    X_valid = train.values[val_idx]
    y_valid =  Y.values[val_idx]
    
    
    lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=train.columns.tolist(),
                         )

    lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=train.columns.tolist(),
                         )

    modelstart = time.time()
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=30000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=200,
        verbose_eval=100
    )
    
    test_pred = lgb_clf.predict(test.values)
    Y_target.append(np.exp(test_pred)-1)
    
    valid_pred = lgb_clf.predict(X_valid)
    
    scores.append(rms(y_valid, valid_pred))
    print('RMSE', rms(y_valid, valid_pred))
    
    print('fold finish after', time.time()-modelstart)

print('Mean RMSE: ', np.mean(scores))
Y_target = np.array(Y_target)

sub = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv')
sub['target'] = Y_target.mean(axis=0)
sub.to_csv('sub_encoded_lgb.csv', index=False)