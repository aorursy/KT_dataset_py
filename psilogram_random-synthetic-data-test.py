# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
#Load training data

data = pd.read_csv('../input/train.csv', index_col=0)

features = [i for i in data.columns if i != 'target']
#Generate fake random data based on train distributions

np.random.seed(1)

new_train = pd.DataFrame()

new_rows = 273000



for f in features:

    std = data[f].std()

    m = data[f].mean()

    

    new_train[f] = np.random.normal(m, std, new_rows)
#Randomly set  3000 rows to target=1, the others to target=0

new_train['target'] = 0

pos_idx = new_train.sample(n=3000, random_state=1).index

new_train.loc[pos_idx, 'target'] = 1
#Now round all features to 4 digits

new_train[features] = new_train[features].round(4)

new_train.head()
#Now upsample positive rows with shuffle method

#Augment 

def augment(x, targ=1, seed=1):

    extra = x[x.target==targ].copy()



    for f in range(200):

        np.random.seed(seed)        

        feat = 'var_' + str(f)

        np.random.shuffle(extra[feat].values)     

        seed +=1

    return extra



extras = []

for i in range(9):

    extra_pos = augment(new_train, targ=1, seed=i)

    extras.append(extra_pos)

    

extras = pd.concat(extras)

new_train = pd.concat((new_train, extras)).reset_index(drop=True)
#Check that pos frequency is about 10%

new_train['target'].value_counts()/len(new_train)
#Remove 100000 rows for test set

new_test = new_train.sample(100000, random_state=1)

new_train = new_train.drop(new_test.index)

print('Train shape:', new_train.shape, 'Test shape:', new_test.shape)
#Check distribution of a favorite variable

new_train[new_train.target==0]['var_81'].hist(bins=100)

new_train[new_train.target==1]['var_81'].hist(bins=100)
#Run 5-fold CV

def get_params(seed=1):

    param = {'num_leaves': 6,

             #'min_data_in_leaf': 20,

             'objective':'binary',

             'metric': 'auc',

             'learning_rate': 0.2,

             "boosting": "gbdt",

             #"feature_fraction": 1.0,

             #"bagging_freq": 5,

             #"bagging_fraction": 0.8,

             "lambda_l2": 10,

             "verbosity": -1,

             "seed": seed,            

            }

    return param



dtrain = lgb.Dataset(new_train[features], new_train.target)

cv = lgb.cv(get_params(), dtrain, 100000, nfold=5, 

            early_stopping_rounds= 100,

            verbose_eval=100

       )



best_score = np.max(cv['auc-mean'])

best_iter = np.argmax(cv['auc-mean'])



print( best_score, best_iter)