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
root = "/kaggle/input/data-science-bowl-2019/"

keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']

train = pd.read_csv(root + 'train.csv',usecols=keep_cols)

test = pd.read_csv(root + 'test.csv', usecols=keep_cols)



train_labels = pd.read_csv(root + 'train_labels.csv')

specs = pd.read_csv(root + 'specs.csv')

sample_submission = pd.read_csv(root + 'sample_submission.csv')
def group_and_reduce(df):

    # group1 and group2 are intermediary "game session" groups,

    # which are reduced to one record by game session. group1 takes

    # the max value of game_time (final game time in a session) and 

    # of event_count (total number of events happened in the session).

    # group2 takes the total number of event_code of each type

    

    # group1 tìm lần chơi cuối cùng của một game_session

    

    group1 = df.drop(columns=['event_id', 'event_code']).groupby(

        ['game_session', 'installation_id', 'title', 'type', 'world']

    ).max().reset_index()  



    # group2 tính tổng các phiên event_code của mỗi installation_id

    group2 = pd.get_dummies(

        df[['installation_id', 'event_code']], 

        columns=['event_code']

    ).groupby(['installation_id']).sum()



    # group3, group4 and group5 are grouped by installation_id 

    # and reduced using summation and other summary stats

    group3 = pd.get_dummies(

        group1.drop(columns=['game_session', 'event_count', 'game_time']),

        columns=['title', 'type', 'world']

    ).groupby(['installation_id']).sum()



    group4 = group1[

        ['installation_id', 'event_count', 'game_time']

    ].groupby(

        ['installation_id']

    ).agg([np.sum, np.mean, np.std])



    return group2.join(group3).join(group4)
train_small = group_and_reduce(train)

test_small = group_and_reduce(test)



print(train_small.shape)

train_small.head()
from sklearn.model_selection import KFold

import lightgbm as lgb

# https://www.kaggle.com/caesarlupum/ds-bowl-start-here-a-gentle-introduction

small_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')

train_joined = train_small.join(small_labels).dropna()

kf = KFold(n_splits=5, random_state=2019)

X = train_joined.drop(columns='accuracy_group').values

y = train_joined['accuracy_group'].values.astype(np.int32)

y_pred = np.zeros((len(test_small), 4))

for train, test in kf.split(X):

    x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]

    train_set = lgb.Dataset(x_train, y_train)

    val_set = lgb.Dataset(x_val, y_val)



    params = {

        'learning_rate': 0.01,

        'bagging_fraction': 0.9,

        'feature_fraction': 0.9,

        'num_leaves': 50,

        'lambda_l1': 0.1,

        'lambda_l2': 1,

        'metric': 'multiclass',

        'objective': 'multiclass',

        'num_classes': 4,

        'random_state': 2019

    }



    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50, valid_sets=[train_set, val_set], verbose_eval=50)

    y_pred += model.predict(test_small)

y_pred
y_pred.argmax(axis=1)
y2 = model.predict(test_small)
y2.argmax(axis=1)
test_small['accuracy_group'] = y2.argmax(axis=1)

test_small[['accuracy_group']].to_csv('submission.csv')