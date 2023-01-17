import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import lightgbm as lgb

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_sample_weight

from scipy import stats



%matplotlib inline
demog = pd.read_csv('../input/tj19data/demo.csv')

demog.isnull().sum() / len(demog)
demog['c1'] = demog['c1'].fillna(0).astype(int)

demog['n2'] = demog['n2'].fillna(0)



tmp = {v: i for i,v in enumerate(sorted(demog['n2'].unique()))}

demog['c_n2'] = demog['n2'].apply(lambda x: tmp[x])



tmp = {v: i for i,v in enumerate(sorted(demog['c2'].unique()))}

demog['c2'] = demog['c2'].apply(lambda x: tmp[x])



demog = demog[['id', 'c0', 'c1', 'c2', 'c3', 'c4', 'n0', 'c_n2']]

demog.head()
txn = pd.read_csv('../input/tj19data/txn.csv')

txn.head()
def mean_filter_zero(series):

    series = np.array(series)

    series = series[series > 0]

    if len(series) == 0:

        return 0

    return np.mean(series)



def min_filter_zero(series):

    series = np.array(series)

    series = series[series > 0]

    if len(series) == 0:

        return 0

    return np.min(series)



def max_filter_zero(series):

    series = np.array(series)

    series = series[series > 0]

    if len(series) == 0:

        return 0

    return np.max(series)



def maximum_valuecounts(series):

    series = np.array(series)

    series = series[series >= 0]

    if len(series) == 0:

        return -1

    _v, _cnt = np.unique(series, return_counts=True)

    return _v[np.argmax(_cnt)]



txn['n4_positive'] = txn['n4'].apply(lambda x: x if x > 0 else 0)

txn['n4_negative'] = txn['n4'].apply(lambda x: -x if x < 0 else 0)

txn['n4_positive_max'] = txn['n4_positive']

txn['n4_negative_max'] = txn['n4_negative']

txn['n4_positive_min'] = txn['n4_positive']

txn['n4_negative_min'] = txn['n4_negative']

txn['n4_max'] = txn['n4']

txn['n4_min'] = txn['n4']

txn['n3_min'] = txn['n3']

txn['n3_max'] = txn['n3']

txn['n5_min'] = txn['n5']

txn['n5_max'] = txn['n5']

txn['n6_min'] = txn['n6']

txn['n6_max'] = txn['n6']
txn_grp = txn.groupby('id').agg({

    'old_cc_label': 'unique',

    'old_cc_no': 'nunique',

    'c5': maximum_valuecounts,

    'c6': maximum_valuecounts,

    'c7': maximum_valuecounts,

    'n3': np.mean,

    'n4': np.mean,

    'n4_positive': mean_filter_zero,

    'n4_negative': mean_filter_zero,

    'n4_positive_min': min_filter_zero,

    'n4_negative_min': min_filter_zero,

    'n4_positive_max': max_filter_zero,

    'n4_negative_max': max_filter_zero,

    'n5': np.mean,

    'n6': mean_filter_zero,

    'n7': 'nunique',

    'n3_max': 'max',

    'n4_max': 'max',

    'n5_max': 'max',

    'n6_max': 'max',

    'n3_min': 'min',

    'n4_min': 'min',

    'n5_min': 'min',

    'n6_min': 'min',

})



for i in range(13):

    txn_grp[f'old_cc_label_{i}'] = txn_grp['old_cc_label'].apply(lambda x: 1 if i in x else 0)



txn_grp['label_cnt'] = txn_grp['old_cc_label'].apply(len)

txn_grp.head()
train_id = pd.read_csv('../input/tj19data/train.csv')



train_df = train_id.merge(demog, how='left', on='id')

train_df = train_df.merge(txn_grp, how='left', on='id')

train_df = train_df.drop(['id', 'old_cc_label', 'n4_positive', 'n4_negative'],axis=1)

train_df.head()
tmp_df = train_df[train_df['label'].isin([6,7,9,0,11])].reset_index(drop=True)

tmp_df_3 = train_df[train_df['label'].isin([11])].reset_index(drop=True)

tmp_df_2 = train_df[train_df['label'].isin([5])].reset_index(drop=True)



train_df_2 = pd.concat([train_df] + [tmp_df] + [tmp_df_2]*300 + [tmp_df_3]*5, sort=False,ignore_index=True)



train_df_2.label.value_counts()
test_id = pd.read_csv('../input/tj19data/test.csv')



test_df = test_id.merge(demog, how='left', on='id')

test_df = test_df.merge(txn_grp, how='left', on='id')

test_df = test_df.drop(['old_cc_label', 'n4_positive', 'n4_negative'],axis=1)

test_df.head()
X = train_df_2.drop(['label'], axis=1)

y = train_df_2['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)



lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=['c0', 'c3', 'c4'])

lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature=['c0', 'c3', 'c4'], reference=lgb_train)
weight_sc = 1 - np.unique(train_id.label.tolist(), return_counts=True)[1] / len(train_id)

def score(y_true, y_pred):

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)

    y_true = np.array([[1 if _y == i else 0 for i in range(13)] for _y in y_true])

    y_pred = np.maximum(1e-80, y_pred)

    score = -1.0 * (weight_sc*(y_true * np.log(y_pred))).sum(axis=1).mean()

    return abs(score)



def techjam_eval(y_pred, dtrain):

    y_true = dtrain.get_label()

    y_pred = y_pred.reshape(13, y_pred.shape[0]//13).T

    return 'techjam_score', score(y_true, y_pred), False



model_params = {

    "objective": 'softmax',

    'boosting_type': 'gbdt',

    'metric': {'multi_logloss'},

    'num_class': 13,

    'num_leaves': 300,

    'learning_rate': 0.005,

    'feature_fraction': 0.8,

    'bagging_fraction': 0.7,

    'bagging_freq': 5,

    'bagging_seed': 42,

    'verbose': 0

}



gbm = lgb.train(model_params,

                lgb_train,

                num_boost_round=3000,

                valid_sets=lgb_eval,

                feval= techjam_eval,

                verbose_eval=100,

                early_stopping_rounds=5)
y_pred = gbm.predict(test_df.drop('id',axis=1))

res_col = ['id'] + [f'class{i}' for i in range(13)]

res = pd.DataFrame([[_id] + [_val for _val in _y] for (_id, _y) in zip(test_df['id'].tolist(), y_pred)], columns=res_col)

res.head()
res.to_csv('res_demog_txn_tune2.csv', index=False)