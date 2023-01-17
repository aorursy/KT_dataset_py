import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
test_df.head()
train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']

test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']



# adding additional features

train_df['hour'] = train_df['DepTime'] // 100

train_df['minute'] = train_df['DepTime'] % 100



test_df['hour'] = test_df['DepTime'] // 100

test_df['minute'] = test_df['DepTime'] % 100



train_df['summer'] = (train_df['Month'].isin(['c-6', 'c-7', 'c-8'])).astype(np.int32)

train_df['autumn'] = (train_df['Month'].isin(['c-9', 'c-10', 'c-11'])).astype(np.int32)

train_df['winter'] = (train_df['Month'].isin(['c-12', 'c-1', 'c-2'])).astype(np.int32)

train_df['spring'] = (train_df['Month'].isin(['c-3', 'c-4', 'c-5'])).astype(np.int32)



test_df['summer'] = (test_df['Month'].isin(['c-6', 'c-7', 'c-8'])).astype(np.int32)

test_df['autumn'] = (test_df['Month'].isin(['c-9', 'c-10', 'c-11'])).astype(np.int32)

test_df['winter'] = (test_df['Month'].isin(['c-12', 'c-1', 'c-2'])).astype(np.int32)

test_df['spring'] = (test_df['Month'].isin(['c-3', 'c-4', 'c-5'])).astype(np.int32)
train_df.head()
categ_feat_idx = np.where(train_df.drop('dep_delayed_15min', axis=1).dtypes == 'object')[0]

categ_feat_idx
X_train = train_df.drop('dep_delayed_15min', axis=1).values

y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

X_test = test_df.values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
ctb = CatBoostClassifier(random_seed=17, silent=True)
%%time

ctb.fit(X_train_part, y_train_part,

        cat_features=categ_feat_idx);
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, ctb_valid_pred)
%%time

ctb.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')
!head ctb_pred.csv