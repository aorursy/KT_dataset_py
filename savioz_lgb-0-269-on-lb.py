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





# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897

def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)

 

def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_lgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return 'gini', gini_score, False





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import sys

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#folder = os.getcwd() + "\\" + "portoseguro\\"





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

from tqdm import tqdm



print('Loading data...')



train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')



print('Train shape:', train.shape)

print('Test shape:', test.shape)



print(train.columns)    



# We drop these variables as we don't want to train on them

# The other 57 columns are all numerical and can be trained on without preprocessing



ids = test['id']



train = train.drop(['id','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin',\

                'ps_ind_13_bin','ps_car_03_cat','ps_car_05_cat'], axis=1)

test = test.drop(['id','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin',\

                'ps_ind_13_bin','ps_car_03_cat','ps_car_05_cat'], axis=1)

                



print('Train shape:', train.shape)

print('Test shape:', test.shape)



print("end loading data...")







X = np.array(train.drop(['target'], axis=1))

y = train['target'].values



X_test = np.array(test)



#Split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, \

    test_size=0.1, random_state = 12)







d_train = lgb.Dataset(X_train, label=y_train)

d_valid = lgb.Dataset(X_valid, label=y_valid) 



watchlist = [d_train, d_valid]







params = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 30,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'max_depth': 15,

    'verbose': 0

}





print('Training LGBM model...')





#default num_boost_round = 200

model = lgb.train(params, train_set=d_train, num_boost_round=100, valid_sets=watchlist, \

early_stopping_rounds=100, verbose_eval=10, feval=gini_lgb)





print("End training")