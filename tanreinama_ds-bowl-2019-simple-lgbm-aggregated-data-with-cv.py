import os



import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
%%time

# Only load those columns in order to save space

keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', usecols=keep_cols)

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', usecols=keep_cols)

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
def group_and_reduce(df):

    # group1 and group2 are intermediary "game session" groups,

    # which are reduced to one record by game session. group1 takes

    # the max value of game_time (final game time in a session) and 

    # of event_count (total number of events happened in the session).

    # group2 takes the total number of event_code of each type

    group1 = df.drop(columns=['event_id', 'event_code']).groupby(

        ['game_session', 'installation_id', 'title', 'type', 'world']

    ).max().reset_index()



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
%%time

train_small = group_and_reduce(train)

test_small = group_and_reduce(test)



print(train_small.shape)

train_small.head()
from sklearn.model_selection import KFold

small_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')

train_joined = train_small.join(small_labels).dropna()

kf = KFold(n_splits=10, random_state=2019)

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

        'num_leaves': 14,

        'lambda_l1': 0.1,

        'lambda_l2': 1,

        'metric': 'multiclass',

        'objective': 'multiclass',

        'num_classes': 4,

        'random_state': 2019

    }



    model = lgb.train(params, train_set, num_boost_round=10000, early_stopping_rounds=300, valid_sets=[train_set, val_set], verbose_eval=100)

    y_pred += model.predict(test_small)
y_pred = y_pred.argmax(axis=1)

test_small['accuracy_group'] = y_pred

test_small[['accuracy_group']].to_csv('submission.csv')