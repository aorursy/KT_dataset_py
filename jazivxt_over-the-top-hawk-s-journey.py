import numpy as np

import pandas as pd

from sklearn import *

import xgboost as xgb



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

train.shape, test.shape, sub.shape
col = [c for c in train.columns if c not in ['id','target']]

def features(df):

    global col

    df['mean'] = df.mean(axis=1)

    df['std'] = df.std(axis=1)

    df['max'] = df.max(axis=1)

    df['min'] = df.min(axis=1)

    df['mad'] = df.mad(axis=1)

    df['kurtosis'] = df.kurtosis(axis=1)

    df['skew'] = df.skew(axis=1)

    

    df['meanc'] = df[col].mean(axis=1)

    df['stdc'] = df[col].std(axis=1)

    df['maxc'] = df[col].max(axis=1)

    df['minc'] = df[col].min(axis=1)

    df['madc'] = df[col].mad(axis=1)

    df['kurtosisc'] = df[col].kurtosis(axis=1)

    df['skewc'] = df[col].skew(axis=1)

    return df



train = features(train)

test = features(test)

train.shape, test.shape
col = [c for c in train.columns if c not in ['id','target']]



fold = 5

test['target'] = 0.0

for i in range(fold):

    model = xgb.XGBClassifier(n_jobs=-1, learning_rate=0.005, n_estimators=5000, seed=i, eval_metric='auc')

    model.fit(train[col], train['target'])

    print(metrics.roc_auc_score(train['target'], model.predict_proba(train[col])[:,1]))

    test['target'] += model.predict_proba(test[col])[:,1]



test['target'] /= fold

test[['id','target']].to_csv('submission.csv', index=False)
train = pd.concat((train,test), sort=False)

print(train.shape)



fold = 5

test['target'] = 0.0

for i in range(fold):

    model = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=30, random_state=i)

    model.fit(train[col], train['target'])

    print(metrics.roc_auc_score(train['target'].round().astype(int), model.predict(train[col])))

    test['target'] += model.predict(test[col])



test['target'] /= fold

test['target'] = test['target'].clip(0.00001, 0.99998)

test[['id','target']].to_csv('submission.csv', index=False)