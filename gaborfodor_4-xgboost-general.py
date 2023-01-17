%matplotlib inline

import os

import pandas as pd

import datetime as dt

import numpy as np

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

from tqdm import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb
plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

pd.set_option('display.max_columns', 99)

start = dt.datetime.now()
full = pd.read_csv('../input/feature-extraction/features_v3.csv.gz')

full.shape
TRAIN_SAMPLE_SIZE = 0.5
full['random'] = np.random.rand(len(full))



train = full[full.IsTrain == 1]

test = full[full.IsTrain == 0]



column_stats = pd.concat([

    pd.DataFrame(full.count()).rename(columns={0: 'cnt'}),

    pd.DataFrame(full.nunique()).rename(columns={0: 'unique'}),

], sort=True, axis=1)

column_stats.sort_values(by='unique')



train_columns = list(column_stats[column_stats.cnt < 10 ** 6].index)

print(train_columns)



target_columns = [

    'TotalTimeStopped_p20',

    'TotalTimeStopped_p50',

    'TotalTimeStopped_p80',

    'DistanceToFirstStop_p20',

    'DistanceToFirstStop_p50',

    'DistanceToFirstStop_p80',

]



do_not_use = train_columns + [

    'IsTrain', 'Path', 'RowId', 'IntersectionId',

    'random', 'intersection_random', 'ValidationGroup',

    'Intersection'

]



feature_columns = [c for c in full.columns if c not in do_not_use]

print(len(feature_columns))

print(feature_columns)
fix = {

    'lambda': 1., 'nthread': 4, 'booster': 'gbtree',

    'silent': 1, 'eval_metric': 'rmse',

    'objective': 'reg:squarederror'}

config = dict(min_child_weight=20,

              eta=0.02, colsample_bytree=0.5,

              max_depth=10, subsample=0.8)

config.update(fix)

nround = 1500
intersections = train.groupby('Intersection')[

    ['RowId']].count().reset_index()

intersections = intersections.drop('RowId', axis=1)

intersections['intersection_random'] = np.random.rand(len(intersections))

train = train.merge(intersections, on='Intersection')
total_mse = 0.0

submission_parts = []

feature_importances = []

for i, target in enumerate(target_columns):

    print(f'Training and predicting for target {target}')

    train_idx = train.intersection_random < TRAIN_SAMPLE_SIZE

    valid_idx = train.intersection_random >= TRAIN_SAMPLE_SIZE



    Xtr = train[train_idx][feature_columns]

    Xv = train[valid_idx][feature_columns]

    ytr = train[train_idx][target].values

    yv = train[valid_idx][target].values

    print(Xtr.shape, ytr.shape, Xv.shape, yv.shape)



    dtrain = xgb.DMatrix(Xtr, label=ytr)

    dvalid = xgb.DMatrix(Xv, label=yv)



    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    model = xgb.train(config, dtrain, nround, evals=watchlist,

                      verbose_eval=100, early_stopping_rounds=50)



    pv = model.predict(dvalid)

    mse = np.mean((yv - pv) ** 2)

    total_mse += mse / 6

    print(target, 'rmse', np.sqrt(mse))

    

    # Feature Importance

    f = model.get_fscore()

    f = pd.DataFrame({'feature': list(f.keys()), 'imp': list(f.values())})

    f.imp = f.imp / f.imp.sum()

    feature_importances.append(f)

    fimp = pd.concat(feature_importances).groupby(['feature']).sum()

    fimp = fimp.sort_values(by='imp', ascending=False).reset_index()

    fimp.to_csv('fimp_general.csv', index=False)



    df = pd.DataFrame({

        'TargetId': test.RowId.astype(str) + '_' + str(i),

        'Target': model.predict(xgb.DMatrix(test[feature_columns]))})

    submission_parts.append(df)
rmse = np.sqrt(total_mse)

print('Total rmse', rmse)

submission = pd.concat(submission_parts, sort=True)

submission.to_csv('general.csv', index=False)
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))