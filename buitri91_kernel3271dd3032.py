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
train = pd.read_csv('/kaggle/input/covid19week3/train.csv')

train.head()
X = train.drop(['confirmed', 'deaths'], axis=1).copy()

yc = train[['confirmed']].copy()

yd = train[['deaths']].copy()

X.shape, yc.shape, yd.shape
test = pd.read_csv('/kaggle/input/covid19week3/test.csv')

test.head()
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')

submission.head()
cparams = {'grow_policy': 'lossguide',

 'learning_rate': 0.01234326155389247,

 'alpha': 0.00010034724805595187,

 'lambda': 0.0018215616133055803,

 'gamma': 0.004947406626842752,

 'max_depth': 29,

 'max_leaves': 232,

 'subsample': 0.5016962178861398,

 'colsample_bytree': 0.9063505797834889,

 'eval_metric': 'rmse',

 'seed': 0}

cparams
ctrain = xgb.DMatrix(X, label=yc)

xtest = xgb.DMatrix(test)



xgc = xgb.train(cparams, ctrain, evals=[(ctrain, 'train')], num_boost_round=10000, early_stopping_rounds=100, verbose_eval=False)

cx = xgc.predict(xtest)

cx[cx < 0] = 0

cx.shape
dparams = {'grow_policy': 'lossguide',

 'learning_rate': 0.01032239445199376,

 'alpha': 1.3211750922259167,

 'lambda': 0.0006262436543771906,

 'gamma': 0.0001686236090992705,

 'max_depth': 18,

 'max_leaves': 202,

 'subsample': 0.5856423840707616,

 'colsample_bytree': 0.8994625185197175,

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