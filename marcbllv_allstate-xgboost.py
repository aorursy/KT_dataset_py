# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
sns.distplot(train.loss)
train.loc[:,'loss'] = np.log(1 + train.loss)

sns.distplot(train.loss)
params = {'eta':0.01, 'nthread':12, 'silent':0, 'eval_metric':'mae',

          'subsample': 0.83, 'colsample_bytree': 0.45, 'gamma': 0.21, 'alpha': 0.24, 

          'max_depth': 7, 'min_child_weight': 2.0, 'lambda': 0.46}

num_boost_round = 1000



tr = train.drop(['id', 'loss'], axis=1).values

lbl = train['loss']



print('Training...')

bst = xgb.train(params=params, 

                dtrain=xgb.DMatrix(tr, label=lbl), 

                num_boost_round=num_boost_round)

print('Done')