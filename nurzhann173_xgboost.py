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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
datatrain = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')

datatest = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')
x = datatrain.iloc[:, 2:].values

y = datatrain.target.values

x_test = datatest.iloc[:, 1:].values
x_train = x

y_train = y
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
datatrain.isnull().values.any()
params = {

    'min_child_weight': 10.0,

    'objective': 'binary:logistic',

    'max_depth': 7,

    'max_delta_step': 1.8,

    'colsample_bytree': 0.4,

    'subsample': 0.8,

    'eta': 0.025,

    'gamma': 0.65,

    'num_boost_round' : 700

    }
X = datatrain.drop(['ID_code', 'target'], axis=1).values

y = datatrain.target.values

test_id = datatest.ID_code.values

test = datatest.drop('ID_code', axis=1)
submission = pd.DataFrame()

submission['ID_code'] = test_id

submission['target'] = np.zeros_like(test_id)

submission.to_csv('submission_xgboost.csv', index=False)