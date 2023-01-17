import os

import numpy as np

import pandas as pd
NFOLDS = 5

np.random.seed(1987)
def angle_to_int(angle):

    a0, a1 = angle.split('.')

    if len(a1) <= 4:

        a1 = (a1 + '0000')[:4]

        return int(a0 + a1)

    else:

        return -999
train = pd.read_json("../input/train.json", dtype={'inc_angle': str})

train['inc_angle'] = train['inc_angle'].replace('na', '-1.0000')

test = pd.read_json("../input/test.json", dtype={'inc_angle': str})

train = train.drop(['band_1', 'band_2', 'is_iceberg'], axis=1)

test = test.drop(['band_1', 'band_2'], axis=1)

train['inc_angle'] = train.inc_angle.apply(angle_to_int)

test['inc_angle'] = test.inc_angle.apply(angle_to_int)

print(train.head(3))

print(test.head(3))
angles = pd.concat([train, test]).groupby('inc_angle').count()[['id']].reset_index()

angles.columns = ['inc_angle', 'cnt']

angles['cv'] = np.random.randint(0, NFOLDS, len(angles))

train = train.merge(angles, on='inc_angle')

test = test.merge(angles, on='inc_angle')

train.loc[train.inc_angle < 0, 'cv'] = np.random.randint(0, NFOLDS, len(train.loc[train.inc_angle < 0]))

test.loc[test.inc_angle < 0, 'cv'] = np.random.randint(0, NFOLDS, len(test.loc[test.inc_angle < 0]))

train.to_csv('train_angle_cv_folds.csv', index=False)

test.to_csv('test_angle_cv_folds.csv', index=False)
print(train.shape)

train.head(10)

print(test.shape)

test.head(10)