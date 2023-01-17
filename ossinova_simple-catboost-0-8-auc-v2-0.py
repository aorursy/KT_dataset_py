import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/cat-in-the-dat/')
train_df = pd.read_csv(PATH_TO_DATA / 'train.csv')
train_df.head()
test_df = pd.read_csv(PATH_TO_DATA / 'test.csv')
test_df.head()
categ_feat_idx = np.where(train_df.drop('target', axis=1).dtypes == 'object')[0]

categ_feat_idx
X_train = train_df.drop('target', axis=1).values

y_train = train_df['target'].values

X_test = test_df.values
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
SEED = 17

params = {'loss_function':'Logloss', # objective function

          'eval_metric':'AUC', # metric

          'verbose': 200, # output to stdout info about training process every 200 iterations

          'early_stopping_rounds': 200,

          'cat_features': categ_feat_idx,

          #'task_type': 'GPU',

          'random_seed': SEED

         }

ctb = CatBoostClassifier(**params)
%%time

ctb.fit(X_train_part, y_train_part,

        eval_set=(X_valid, y_valid),

        use_best_model=True,

        plot=True);
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, ctb_valid_pred)
%%time

ctb.fit(X_train, y_train,

        eval_set=(X_valid, y_valid),

        use_best_model=True,

        plot=True);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['target'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')
!head ctb_pred.csv