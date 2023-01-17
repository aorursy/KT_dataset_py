import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

import warnings

warnings.filterwarnings('ignore')
import os

os.listdir('../input/')
train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv') 

test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv') 
# train_stat = pd.read_csv('../input/stats-features/train_stat.csv') 

# test_stat = pd.read_csv('../input/stats-features/test_stat.csv') 
# train_stat['ID_code'] = train_df.ID_code

# test_stat['ID_code'] = test_df.ID_code
# train = pd.merge(train_df, train_stat,on='ID_code')

# test = pd.merge(test_df, test_stat,on='ID_code')
train = train_df.copy()

test = test_df.copy()
del train_df

# del train_stat

del test_df

# del test_stat
train.shape, test.shape
def augment(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y

train.head()
test.head()
features = [c for c in train.columns if c not in ['ID_code', 'target']]

target = train['target']

X_test = test[features].values
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.335,

    'boost_from_average':'false',

    'boost': 'gbdt',

    'feature_fraction': 0.041,

    'learning_rate': 0.0083,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 80,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1}
num_folds = 9

features = [c for c in train.columns if c not in ['ID_code', 'target']]



folds = KFold(n_splits=num_folds, random_state=2319)

oof = np.zeros(len(train))

getVal = np.zeros(len(train))

predictions = np.zeros(len(target))

feature_importance_df = pd.DataFrame()



print('Light GBM Model')

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    

    X_train, y_train = train.iloc[trn_idx][features], target.iloc[trn_idx]

    X_valid, y_valid = train.iloc[val_idx][features], target.iloc[val_idx]

    

    X_tr, y_tr = augment(X_train.values, y_train.values)

    X_tr = pd.DataFrame(X_tr)

    

    print("Fold:{}".format(fold_ + 1))

    trn_data = lgb.Dataset(X_tr, label=y_tr)

    val_data = lgb.Dataset(X_valid, label=y_valid)

    val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx])

    

    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=5000, early_stopping_rounds = 4000)

    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

    getVal[val_idx]+= clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration) / folds.n_splits

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits



print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))
predictions['target'] = np.mean(predictions[[col for col in predictions.columns if col not in ['ID_code', 'target']]].values, axis=1)

sub = pd.DataFrame({"ID_code": test.ID_code.values})

sub["target"] = predictions['target']

sub.to_csv('submission_oof.csv', index=False)
sub.head(2)