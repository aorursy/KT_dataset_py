import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/focus-start-2020')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
categ_feat_idx = np.where(train_df.drop('dep_delayed_15min', axis=1).dtypes == 'object')[0]

categ_feat_idx
X_train = train_df.drop('dep_delayed_15min', axis=1).values

y_train = train_df['dep_delayed_15min']

X_test = test_df.values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=42)
ctb = CatBoostClassifier(random_seed=42, silent=True)
%%time

ctb.fit(X_train_part, y_train_part, cat_features=categ_feat_idx)
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]

print(roc_auc_score(y_valid, ctb_valid_pred))
%%time

ctb.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv')
sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv')

sample_sub['Predicted'] = ctb_test_pred

sample_sub.to_csv('ctb_pred.csv', index=False)
!head ctb_pred.csv