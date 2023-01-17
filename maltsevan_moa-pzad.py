# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss

from sklearn.multioutput import MultiOutputClassifier

from sklearn.multiclass import OneVsRestClassifier

from category_encoders import CountEncoder

from xgboost import XGBClassifier

from catboost import Pool, CatBoostClassifier

from lightgbm import LGBMClassifier



import matplotlib.pyplot as plt

import seaborn as sns

import os
SEED = 42

NFOLDS = 5

DATA_DIR = '/kaggle/input/lish-moa/'

np.random.seed(SEED)
train = pd.read_csv(DATA_DIR + 'train_features.csv')

targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')



test = pd.read_csv(DATA_DIR + 'test_features.csv')

sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')



X = train.iloc[:,1:]

X_test = test.iloc[:,1:]

y = targets.iloc[:,1:] 
stats_unique_values = []

for col in X:

    stats_unique_values.append([col, train[col].nunique()])

stats_unique_values = pd.DataFrame(stats_unique_values, columns=['column', 'count_unique'])    
sns.distplot(stats_unique_values['count_unique'])
display(stats_unique_values[stats_unique_values['count_unique'] < 100])

categ_columns = list(stats_unique_values.loc[stats_unique_values['count_unique'] < 100, 'column'])
classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))



clf = Pipeline([('encode', CountEncoder(cols=categ_columns)),

                ('classify', classifier)

               ])

oof_preds = np.zeros(y.shape)

test_preds = np.zeros((test.shape[0], y.shape[1]))

oof_losses = []

kf = KFold(n_splits=NFOLDS)



for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):

    print(fn)

    X_train, X_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]

    y_train, y_val = y.iloc[trn_idx, :], y.iloc[val_idx, :]

    

    ctl_mask = X_train.iloc[:,0]=='ctl_vehicle'

    X_train = X_train.loc[~ctl_mask,:]

    y_train = y_train.loc[~ctl_mask, :]

    

    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val)

    val_preds = np.array(val_preds)[:,:,1].T

    oof_preds[val_idx] = val_preds

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T

    test_preds += preds / NFOLDS

    

print(oof_losses)

print('Mean loss ', np.mean(oof_losses))

print('STD loss ', np.std(oof_losses))
control_mask = train['cp_type']=='ctl_vehicle'

oof_preds[control_mask] = 0



print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))



control_mask = test['cp_type']=='ctl_vehicle'

test_preds[control_mask] = 0



sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)
!ls