import numpy as np 

import pandas as pd

from sklearn import *

import lightgbm as lgb



train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

train.shape, test.shape
def feature(df):

    df.index = (df.time*10000 - 1).values

    df['batch'] = df.index // 25000 

    df['mean'] = df.groupby('batch')['signal'].mean()

    df['median'] = df.groupby('batch')['signal'].median()

    df['max'] = df.groupby('batch')['signal'].max()

    df['min'] = df.groupby('batch')['signal'].min()

    df['std'] = df.groupby('batch')['signal'].std()

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    return df
train = feature(train)

test = feature(test)
train.head()
col = [c for c in train.columns if c not in ['time', 'open_channels']]

x1, x2, y1, y2 = model_selection.train_test_split(train[col], train['open_channels'], test_size=0.3, random_state=7)

del train
def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = metrics.f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)
params = {'learning_rate': 0.1, 

          'max_depth': -1, 

          'num_leaves': 200,

          'metric': 'logloss', 

          'random_state': 7, 

          'n_jobs':-1, 

          'sample_fraction':0.33}

# model = lgb.train(params, lgb.Dataset(x1, y1), 2000, lgb.Dataset(x2, y2), verbose_eval=25, early_stopping_rounds=200, feval=MacroF1Metric)
preds = model.predict(test[col], num_iteration=model.best_iteration)

test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)
len(test)
test[['time','open_channels']].to_csv('submission1.csv', index=False, float_format='%.4f')