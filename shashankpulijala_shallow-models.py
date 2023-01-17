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
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

train_targets_scored = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')
X = train_features.iloc[:,1:].to_numpy()

y = train_targets_scored.iloc[:, 1:].to_numpy()

X_test = test_features.iloc[:,1:].to_numpy()

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss



import matplotlib.pyplot as plt



from sklearn.multioutput import MultiOutputClassifier



import os

import warnings

warnings.filterwarnings('ignore')
np.random.seed(64)

NFOLDS = 5
from sklearn.ensemble import RandomForestClassifier

classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))

clf = Pipeline([('encoder' , CountEncoder(cols = [0, 2])),

                ('classify' , classifier)])
params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
# from sklearn.model_selection import train_test_split

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state = 64)

# ctl_mask = X_train[:,0]=='ctl_vehicle'

# X_train  = X_train[~ctl_mask, :]

# y_train = y_train[~ctl_mask]

# clf.fit(X_train, y_train)

# val_preds = clf.predict_proba(X_val)

# val_preds
# np.array(val_preds)[:,:,1].shape
# np.array(val_preds).shape
#kfold 

oof_preds = np.zeros(y.shape)

test_preds  = np.zeros((test_features.shape[0], y.shape[1]))

oof_losses = []

kf = KFold(n_splits=NFOLDS)

for fn, (train_idx, valid_idx) in enumerate(kf.split(X, y)) :

    print('Starting fold: ', fn)

    X_train, X_val = X[train_idx], X[valid_idx]

    y_train, y_val = y[train_idx], y[valid_idx]

    #drop ctl vehicle

    ctl_mask = X_train[:,0]=='ctl_vehicle'

    X_train  = X_train[~ctl_mask, :]

    y_train = y_train[~ctl_mask]

    

    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val)

    val_preds  = np.array(val_preds)[:,:,1].T #Taking the pos class

    oof_preds[valid_idx] = val_preds

    loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

    oof_losses.append(loss)

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds += preds / NFOLDS

        

print(oof_losses)

print('Mean OOF loss across folds', np.mean(oof_losses))

print('STD OOF loss across folds', np.std(oof_losses))

    

    



    
control_mask = train_features['cp_type'] == 'ctl_vehicle'

oof_preds[control_mask] = 0 

print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))

control_mask = test_features['cp_type']=='ctl_vehicle'



test_preds[control_mask] = 0
sub =pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
# create the submission file

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)