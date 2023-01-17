# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import xgboost as xgb
train = pd.read_csv('/kaggle/input/covid19week4/train.csv')

train.head()
X = train.drop(['confirmed', 'deaths'], axis=1).copy()

yc = train[['confirmed']].copy()

yd = train[['deaths']].copy()

X.shape, yc.shape, yd.shape
test = pd.read_csv('/kaggle/input/covid19week4/test.csv')

test.head()
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')

submission.head()
cparams = {'grow_policy': 'lossguide',

 'learning_rate': 0.024338128795802872,

 'alpha': 0.003246952885448336,

 'lambda': 0.02492403000653683,

 'gamma': 0.06685904741133017,

 'max_depth': 21,

 'max_leaves': 92,

 'subsample': 0.557513099089599,

 'colsample_bytree': 0.9237395981525455,

 'eval_metric': 'rmse',

 'seed': 0}

cparams
ctrain = xgb.DMatrix(X, label=yc)

xtest = xgb.DMatrix(test)



xgc = xgb.train(cparams, ctrain, evals=[(ctrain, 'train')], num_boost_round=10000, early_stopping_rounds=100, verbose_eval=False)

cx = xgc.predict(xtest)

cx[cx < 0] = 0

cx.shape
dparams = {'grow_policy': 'depthwise',

 'learning_rate': 0.01903207407409567,

 'alpha': 0.00025755625853743136,

 'lambda': 0.0006783670643399993,

 'gamma': 0.40781406773166995,

 'max_depth': 8,

 'max_leaves': 9,

 'subsample': 0.603094706979611,

 'colsample_bytree': 0.8980489190292558,

 'eval_metric': 'rmse',

 'seed': 0}

dparams
dtrain = xgb.DMatrix(X, label=yd)

xtest = xgb.DMatrix(test)



xgd = xgb.train(dparams, dtrain, evals=[(dtrain, 'train')], num_boost_round=10000, early_stopping_rounds=100, verbose_eval=False)

dx = xgd.predict(xtest)

dx[dx < 0] = 0

dx.shape
submission['ConfirmedCases'] = cx

submission['Fatalities'] = dx

submission.head()
submission['ConfirmedCases'] = submission.ConfirmedCases.astype(int)

submission['Fatalities'] = submission.Fatalities.astype(int)

submission.head()
submission.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv')