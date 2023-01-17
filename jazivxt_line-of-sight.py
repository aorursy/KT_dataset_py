import numpy as np

import pandas as pd

from sklearn import *



xtrain = pd.read_csv('../input/X_train.csv')

ytrain = pd.read_csv('../input/y_train.csv')

train = pd.merge(xtrain, ytrain, how='left', on='series_id')



xtest = pd.read_csv('../input/X_test.csv')

ytest = pd.read_csv('../input/sample_submission.csv')

test = pd.merge(xtest, ytest, how='left', on='series_id')



print(train.shape, test.shape)
def features(df):

    for c in ['angular_velocity_', 'linear_acceleration_']:

        col = [c + c1 for c1 in ['X','Y','Z']]

        for agg in ['min(', 'max(', 'sum(', 'mean(', 'std(', 'skew(', 'kurtosis(', 'quantile(.25,', 'quantile(.5,', 'quantile(.75,']:

            df[c+agg] = eval('df[col].' + agg + 'axis=1)')

            df[c+'a'+agg] = eval('df[col].abs().' + agg + 'axis=1)')

    return df



train = features(train).fillna(-999)

test = features(test).fillna(-999)

print(train.shape, test.shape)
col = [c for c in train.columns if c not in ['row_id', 'series_id', 'measurement_number', 'group_id', 'surface']]

le = preprocessing.LabelEncoder()

train['surface'] = le.fit_transform(train['surface'])



split = 387680

clf1 = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=20)

clf2 = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=20)

clf1.fit(train[col][:split], train['surface'][:split])

clf2.fit(train[col][:split], train['surface'][:split])



sub = clf1.predict_proba(train[col][split:])

sub += clf2.predict_proba(train[col][split:])

sub = sub.argmax(axis=1)

print(metrics.accuracy_score(train['surface'][split:], sub))



clf1.fit(train[col], train['surface'])

clf2.fit(train[col], train['surface'])



sub = clf1.predict_proba(test[col])

sub += clf2.predict_proba(test[col])



sub = pd.DataFrame(sub, columns=le.classes_)

sub['series_id'] = test['series_id']

sub = sub.groupby(by=['series_id'], as_index=False).sum()

sub['surface'] = le.inverse_transform(sub[le.classes_].values.argmax(axis=1))

sub[['series_id', 'surface']].to_csv('submission.csv', index=False)