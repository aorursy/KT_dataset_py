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
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

train_features = pd.read_csv('../input/lish-moa/train_features.csv')

sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_targets_scored.head()
train_targets_nonscored.head()
train_targets_nonscored.shape
train_features.head()
train_features.shape
train = train_features.merge(train_targets_scored, on='sig_id')
train_targets_scored.columns
target_cols = [c for c in train_targets_scored.columns if c != 'sig_id']
cols = target_cols + ['cp_type']
train[cols]
train[cols].groupby('cp_type').sum().sum(1)
train[train['cp_type'] == 'ctl_vehicle']
print(train.shape, test_features.shape)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss

from sklearn.model_selection import train_test_split





import matplotlib.pyplot as plt



from sklearn.multioutput import MultiOutputClassifier



import os

import warnings

warnings.filterwarnings('ignore')
train_features.head(1)
X = train_features.iloc[:,1:].to_numpy()

y = train_targets_scored.iloc[:,1:].to_numpy()
# seed = 7

# test_size = 0.30

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# X_train_encode = pd.get_dummies(X_train).to_numpy() 

# X_test_encode = pd.get_dummies(X_test).to_numpy() 

# y_train = y_train.to_numpy() 

# y_test = y_test.to_numpy() 
classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))
clf = Pipeline([('encode', CountEncoder(cols=[0, 2])),

                ('classify', classifier)

               ])
clf
params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
oof_preds = np.zeros(y.shape)
test_features.shape
X_test = test_features.iloc[:,1:].to_numpy()
test_preds = np.zeros((test_features.shape[0], y.shape[1]))

NFOLDS = 2

oof_losses = []

kf = KFold(n_splits=NFOLDS)
for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):

    X_train, X_val = X[trn_idx], X[val_idx]

    y_train, y_val = y[trn_idx], y[val_idx]



    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val)

    val_preds = np.array(val_preds)[:,:,1].T

    oof_preds[val_idx] = val_preds

    

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds += preds / NFOLDS

    

print(oof_losses)

print('Mean OOF loss across folds', np.mean(oof_losses))

print('STD OOF loss across folds', np.std(oof_losses))    
test_preds.shape
sample_submission.shape
# create the submission file

sample_submission.iloc[:,1:] = test_preds

sample_submission.to_csv('submission.csv', index=False)
print(test_features.shape, X_test.shape, X_val.shape, oof_preds.shape, oof_preds[val_idx].shape,test_preds.shape)
np.array(val_preds).shape
val_preds
y[val_idx].shape