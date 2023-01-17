import numpy as np

import pandas as pd

import gc

# importing garbage collector to keep our RAM usgae in check



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import lightgbm as lgbm

train = pd.read_pickle('/kaggle/input/pickle-ieee/Train.pkl')

test = pd.read_pickle('/kaggle/input/pickle-ieee/Test.pkl')

y = train['isFraud']

del train['isFraud']

params = {

 'reg_lambda': 0.1,

 'reg_alpha': 0.1,

 'num_leaves': 800,

 'min_data_in_leaf': 100,

 'learning_rate': 0.05,

 'feature_fraction': 0.4,

 'bagging_fraction': 0.1,

 'verbosity' : -1,

  'objective' : 'binary',

  'random_state' : 42,

  'metric' : 'auc',

  'max_depth' : -1,

  'boosting_type': 'gbdt',

}

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score
n_folds = 5

folds = KFold(n_splits=n_folds)

columns = train.columns

y_preds = np.zeros(test.shape[0])

for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y)):

    X_train, X_valid = train[columns].iloc[train_index], train[columns].iloc[valid_index]

    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    

    temp_train = lgbm.Dataset(X_train, label=y_train)

    temp_valid = lgbm.Dataset(X_valid, label=y_valid)

    clf = lgbm.train(params,temp_train, 10000, valid_sets = [temp_train, temp_valid],

                      verbose_eval=200, early_stopping_rounds=500)

    

    y_pred_valid = clf.predict(X_valid)

    print("AUC: ",roc_auc_score(y_valid, y_pred_valid))

    y_preds += clf.predict(test) / n_folds

    

    del X_train, X_valid, y_train, y_valid

    gc.collect()
submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

submission['isFraud'] = y_preds

submission.to_csv('submission.csv', index=False)