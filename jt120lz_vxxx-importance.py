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

import multiprocessing

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

from sklearn import metrics

import lightgbm as lgb



warnings.simplefilter('ignore')

sns.set()

%matplotlib inline


files = ['../input/ieee-fraud-detection/test_identity.csv', 

         '../input/ieee-fraud-detection/test_transaction.csv',

         '../input/ieee-fraud-detection/train_identity.csv',

         '../input/ieee-fraud-detection/train_transaction.csv',

         '../input/ieee-fraud-detection/sample_submission.csv']
%%time

def load_data(file):

    return pd.read_csv(file)



with multiprocessing.Pool() as pool:

    test_id, test_tr, train_id, train_tr, sub = pool.map(load_data, files)
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')



del test_id, test_tr, train_id, train_tr

gc.collect()


lgb_param = {

    'objective': 'binary',

    "metric": 'auc',

    "verbosity": -1,

    'n_estimators': 10,

    'n_jobs': -1,

    'seed': 2019

}



for i in range(32):

    cols = ['V'+str(i) for i in range(i*10+1, 10*(i+1)+1)]

    print(cols)

    a = train[cols+['isFraud']]

    X = a.iloc[:, :-1]

    y = a.iloc[:, -1]

    split = int(0.8*len(X))

    train_x, train_y, test_x, test_y = X.iloc[:split], y[:split], X.iloc[split:], y[split:]

    model = lgb.LGBMClassifier(**lgb_param)

    model.fit(train_x, train_y)

    pred = model.predict_proba(test_x)[:,1]

    print(metrics.roc_auc_score(test_y, pred))

    t = pd.DataFrame({'name':a.columns[:-1], 'score':model.feature_importances_})

    t = t.sort_values(['score'], ascending=False)

    print(t)

    print('='*20)

#     break