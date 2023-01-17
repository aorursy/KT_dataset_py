import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
def shuffle_col_vals(x1):

    rand_x = np.array([np.random.choice(x1.shape[0], size=x1.shape[0], replace=False) for i in range(x1.shape[1])]).T

    grid = np.indices(x1.shape)

    rand_y = grid[1]

    return x1[(rand_x, rand_y)]



def augment_fast1(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        x1 = shuffle_col_vals(x1)

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        x1 = shuffle_col_vals(x1)

        xn.append(x1)



    xs = np.vstack(xs); xn = np.vstack(xn)

    ys = np.ones(xs.shape[0]);yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn]); y = np.concatenate([y,ys,yn])

    return x,y



def disarrange(a, axis=-1):

    """

    Shuffle `a` in-place along the given axis.



    Apply numpy.random.shuffle to the given axis of `a`.

    Each one-dimensional slice is shuffled independently.

    """

    b = a.swapaxes(axis, -1)

    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,

    # so `a` is shuffled in place, too.

    shp = b.shape[:-1]

    for ndx in np.ndindex(shp):

        np.random.shuffle(b[ndx])

    return



def augment_fast2(x,y,t=2):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        disarrange(x1,axis=0)

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        disarrange(x1,axis=0)

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
%%time

x = train[train.columns[2:]].values

y = train['target'].values

x_ts = test[test.columns[1:]].values
%%time

x1,y1 = augment_fast2(x,y,t=10)

print(x.shape,y.shape,x1.shape,y1.shape)
import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
x_train,x_test,y_train,y_test = train_test_split(x1,y1, test_size = 1/9, random_state=1999111000)
param = {

    'bagging_freq': 5,          

    'bagging_fraction': 0.331,

    'boost_from_average':'false',   

    'boost': 'gbdt',

    'feature_fraction': 0.0405,

    'learning_rate': 0.01,

    'max_depth': -1, 

    'metric':'auc',    

    'min_data_in_leaf': 80, 

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,   

    'num_threads': 8,   

    'tree_learner': 'serial', 

    'objective': 'binary',  

    'verbosity': 1

}
trn_data = lgb.Dataset(x_train, label=y_train)

val_data = lgb.Dataset(x_test, label=y_test)
clf = lgb.train(param, trn_data, num_boost_round=30000, valid_sets = [trn_data, val_data], verbose_eval=2000, early_stopping_rounds = 3500)
predictions = clf.predict(x_ts, num_iteration=clf.best_iteration)
# folds = KFold(n_splits=9, random_state=1999111000)

# predictions = np.zeros(200000)



# for fold_, (trn_idx, val_idx) in enumerate(folds.split(x1, y1)):

    

#     X_train, y_train = x1[trn_idx], y1[trn_idx]

#     X_valid, y_valid = x1[val_idx], y1[val_idx]

    

#     print("Fold idx:{}".format(fold_ + 1))

#     trn_data = lgb.Dataset(X_train, label=y_train)

#     val_data = lgb.Dataset(X_valid, label=y_valid)

    

#     clf = lgb.train(param, trn_data, num_boost_round=20000, valid_sets = [trn_data, val_data], verbose_eval=2000, early_stopping_rounds = 3500)

    

#     predictions += clf.predict(x_ts, num_iteration=clf.best_iteration) / folds.n_splits
sub = pd.read_csv('../input/sample_submission.csv')

sub["target"] = predictions

sub.to_csv("submission.csv", index=False)

sub.head()