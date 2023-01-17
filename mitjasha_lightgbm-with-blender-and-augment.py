ver = 'lgbm_v8'

import warnings

warnings.filterwarnings('ignore')

import time

from datetime import datetime



import numpy as np 

import pandas as pd 



import lightgbm as lgb

from sklearn.model_selection import cross_val_score, train_test_split

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.describe()
# augment from https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

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
#Inspired by Gabriel Preda 's Kernel'

%time

idx = features = train.columns.values[2:202]

for i,df in enumerate([train, test]):

    df['sum'] = df[idx].sum(axis=1)  

    df['min'] = df[idx].min(axis=1)

    df['max'] = df[idx].max(axis=1)

    df['mean'] = df[idx].mean(axis=1)

    df['std'] = df[idx].std(axis=1)

    df['skew'] = df[idx].skew(axis=1)

    df['kurt'] = df[idx].kurtosis(axis=1)

    df['med'] = df[idx].median(axis=1)

    print('Creating percentiles features for df: {}/{}'.format(i+1,2))

    df['perc_5'] =  df[idx].apply(lambda x: np.percentile(x, 5), axis=1)

    df['perc_10'] =  df[idx].apply(lambda x: np.percentile(x, 10), axis=1)

    df['perc_25'] =  df[idx].apply(lambda x: np.percentile(x, 25), axis=1)

    df['perc_50'] =  df[idx].apply(lambda x: np.percentile(x, 50), axis=1)

    df['perc_75'] =  df[idx].apply(lambda x: np.percentile(x, 75), axis=1)

    df['perc_95'] =  df[idx].apply(lambda x: np.percentile(x, 95), axis=1)

    df['perc_99'] =  df[idx].apply(lambda x: np.percentile(x, 99), axis=1)
display(train.head())

display(test.head())
X = train.iloc[:,2:].values

y = train.iloc[:,1].values

test = test.iloc[:,1:].values
lgb.train()
pred = pd.DataFrame()

for i in range (1, 5):

    param = {

        'bagging_freq': 5,

        'bagging_fraction': 0.4,

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.05,

        'learning_rate': 0.01,

        'max_depth': -1,  

        'metric':'auc',

        'min_data_in_leaf': 80,

        'min_sum_hessian_in_leaf': 10.0,

        'num_leaves': 13,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1,

        'seed': i,

        'feature_fraction_seed': i,

        'bagging_seed': i,

        'drop_seed': i,

        'data_random_seed': i,

    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = i)

    print(y_train)

    N = 5

    p_valid,yp = 0,0

    for i in range(N):

        X_t, y_t = augment(X_train, y_train)

        X_t = pd.DataFrame(X_t)

        X_t = X_t.add_prefix('var_')

    

        trn_data = lgb.Dataset(X_t, label=y_t)

        val_data = lgb.Dataset(X_test, label=y_test)

        clf = lgb.train(param, trn_data , 10000, valid_sets = [trn_data, val_data], 

                        verbose_eval=1000, early_stopping_rounds = 100)

        p_valid += clf.predict(X_test)

        yp += clf.predict(test)

    

    print(yp/N)

    pred[i] = yp/N
pred.head()
filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))

filename
submission_ = pd.read_csv('../input/sample_submission.csv')

submission_['target'] = pred.mean(axis=1)

submission_.to_csv(filename+'_blend.csv', index=False)