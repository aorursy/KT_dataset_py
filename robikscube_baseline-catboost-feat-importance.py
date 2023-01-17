import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from catboost import CatBoostClassifier, Pool, cv
# Categorical Features - Transaction

test_identity = pd.read_csv('../input/test_identity.csv')

test_transaction = pd.read_csv('../input/test_transaction.csv')

train_identity = pd.read_csv('../input/train_identity.csv')

train_transaction = pd.read_csv('../input/train_transaction.csv')

ss = pd.read_csv('../input/sample_submission.csv')
FEATURES = [f for f in train_transaction.columns if f not in ['TransactionID','isFraud','TransactionDT']]

CAT_FEATS = train_transaction.select_dtypes('object').columns

# Fill NA Categorical Features

train_transaction[CAT_FEATS] = train_transaction[CAT_FEATS].fillna('_NA_')

test_transaction[CAT_FEATS] = test_transaction[CAT_FEATS].fillna('_NA_')

X = train_transaction[FEATURES]

y = train_transaction['isFraud']

X_test = test_transaction[FEATURES]
train_dataset = Pool(data=X,

                  label=y,

                  cat_features=CAT_FEATS)

# valid_dataset = Pool(data=X_valid,

#                   label=y_valid,

#                   cat_features=CAT_FEATS)

test_dataset = Pool(data=X_test,

                    cat_features=CAT_FEATS)
ITERATIONS = 800



clf = CatBoostClassifier(iterations=ITERATIONS,

                         learning_rate=0.1,

                         depth=15,

                         eval_metric='AUC',

                         random_seed = 529,

                         task_type="GPU",

                         verbose=50)



_ = clf.fit(train_dataset) #, eval_set=valid_dataset)
train_preds = clf.predict_proba(train_dataset)[:,1]

#valid_preds = clf.predict_proba(valid_dataset)[:,1]

test_preds = clf.predict_proba(test_dataset)[:,1]
ss['isFraud'] = test_preds

ss.to_csv('predictions.csv', index=False)
ss.head()
fi = pd.DataFrame(index=clf.feature_names_)

fi['importance'] = clf.feature_importances_

fi.loc[fi['importance'] > 0.1].sort_values('importance').plot(kind='barh', figsize=(15, 25), title='Feature Importance')

plt.show()