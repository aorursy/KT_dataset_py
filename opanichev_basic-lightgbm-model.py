import numpy as np

import os

import pandas as pd

from tqdm import tqdm_notebook
from sklearn.model_selection import KFold

import lightgbm as lgb
data_path = '../input/'

train = pd.read_csv(os.path.join(data_path, 'train.csv'))

test = pd.read_csv(os.path.join(data_path, 'test.csv'))
train.head()
train.describe()
train.shape
test.head()
test.shape
def rmsle(y_true, p): 

    """

    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    

    Args:

        y_true - numpy array containing targets with shape (n_samples, n_targets)

        p - numpy array containing predictions with shape (n_samples, n_targets)

    """

    return np.sqrt(np.square(np.log(p + 1) - np.log(y_true + 1)).mean())
X = np.array(train.drop(['id', 'formation_energy_ev_natom', 'bandgap_energy_ev'], axis=1))

y = np.array(train[['formation_energy_ev_natom', 'bandgap_energy_ev']])



X_test = np.array(test.drop(['id'], axis=1))



print(X.shape, y.shape)

print(X_test.shape)
p_buf = []



mean_rmsle = [0, 0]

for target_i in range(2):

    preds_buf = []

    rmsle_buf = []

    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    for train_index, test_index in kf.split(X):

        y_pred = []

        

        X_train, X_valid = X[train_index], X[test_index]

        y_train, y_valid = y[train_index, target_i], y[test_index, target_i]



        # LGB

        lgb_train = lgb.Dataset(X_train, y_train)

        lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    

        params = {

            'boosting_type': 'gbdt',

            'objective': 'regression',

            'metric': {'rmse', 'rmsle'},

            'max_depth': 8,

            'num_leaves': 31,

            'learning_rate': 0.025,

            'feature_fraction': 0.9,

            'bagging_fraction': 0.8,

            'bagging_freq': 5,

            'verbose': 0,

        }

        

        gbm = lgb.train(params,

                lgb_train,

                num_boost_round=10000,

                valid_sets=[lgb_valid],

                early_stopping_rounds=100,

                verbose_eval=0)

        

        y_pred.append(gbm.predict(X_valid, num_iteration=gbm.best_iteration))

        e = rmsle(y_valid, y_pred[-1])

        rmsle_buf.append(e)

        

        print(target_i, 'lgb', e)        

        p = gbm.predict(X_test, num_iteration=gbm.best_iteration)

        preds_buf.append(p)

        

    p_buf.append(np.mean(preds_buf, axis=0))

    mean_rmsle[target_i] = np.mean(rmsle_buf)

    

print(mean_rmsle)

print(np.mean(mean_rmsle))
subm = pd.DataFrame()

subm['id'] = test['id'].values

subm['formation_energy_ev_natom'] = p_buf[0]

subm['bandgap_energy_ev'] = p_buf[1]

subm.to_csv('submission.csv', index=False)