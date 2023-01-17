# https://github.com/KirillTushin/target_encoding

import os

os.chdir('../input/target-encoding')

from target_encoding import TargetEncoderClassifier, TargetEncoder

os.chdir('/kaggle/working')



import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, cross_val_score
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



    return group2.join(group3).join(group4).reset_index()
%%time

train = group_and_reduce(train)

test = group_and_reduce(test)



print(train.shape)

train.head()
labels = train_labels[['installation_id', 'accuracy_group']]

train = train.merge(labels, how='left', on='installation_id').dropna()
x_train, x_val, y_train, y_val = train_test_split(

    train.drop(['installation_id', 'accuracy_group'], axis=1),

    train['accuracy_group'],

    test_size=0.15,

    random_state=2019,

)
len_uniques = []

train_labeled = train.fillna(-999)

test_labeled = test.fillna(-999)



for c in train.columns.drop(['installation_id', 'accuracy_group']):

    le = LabelEncoder()

    le.fit(pd.concat([train_labeled[c], test_labeled[c]])) 

    train_labeled[c] = le.transform(train_labeled[c])

    test_labeled[c] = le.transform(test_labeled[c])

    len_uniques.append(len(le.classes_))



x_train_labeled, x_val_labeled = train_test_split(

    train_labeled.drop(['installation_id', 'accuracy_group'], axis=1),

    test_size=0.15,

    random_state=2019,

)
ALPHA = 10

MAX_UNIQUE = 50

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)





'''

split: list of int or cross-validator class,

            if split is [], then algorithm will encode features without cross-validation

            This situation features will overfit on target



            if split len is 1 for example [5], algorithm will encode features by using cross-validation on 5 folds

            This situation you will not overfit on tests, but when you will validate, your score will overfit



            if split len is 2 for example [5, 3], algorithm will separate data on 5 folds, afterwords

            will encode features by using cross-validation on 3 folds

            This situation is the best way to avoid overfit, but algorithm will use small data for encode.

'''





enc = TargetEncoder(alpha=ALPHA, max_unique=MAX_UNIQUE, split=[cv])

x_train_encoded = enc.transform_train(x_train_labeled, y=y_train)

x_val_encoded = enc.transform_test(x_val_labeled)

x_test_encoded = enc.transform_test(test.drop(['installation_id'], axis=1))



x_train_encoded = pd.DataFrame(x_train_encoded)

x_val_encoded = pd.DataFrame(x_val_encoded)

x_test_encoded = pd.DataFrame(x_test_encoded)
x_train_all = pd.concat([x_train.reset_index(drop=True), x_train_encoded], axis=1)

x_val_all = pd.concat([x_val.reset_index(drop=True), x_val_encoded], axis=1)

x_test_all = pd.concat([test.drop(['installation_id'], axis=1), x_test_encoded], axis=1)
train_set = lgb.Dataset(x_train_all, y_train)

val_set = lgb.Dataset(x_val_all, y_val)



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
val_pred = model.predict(x_val_all).argmax(axis=1)

print(classification_report(y_val, val_pred))
y_pred = model.predict(x_test_all).argmax(axis=1)

test['accuracy_group'] = y_pred

test[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)