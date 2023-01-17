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
SEED = 19

NFOLDS = 5

DATA_DIR = '/kaggle/input/lish-moa/'

np.random.seed(SEED)
train = pd.read_csv(DATA_DIR + 'train_features.csv')

targets = pd.read_csv(DATA_DIR + 'train_targets_scored.csv')



test = pd.read_csv(DATA_DIR + 'test_features.csv')

sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')



# drop where cp_type==ctl_vehicle (baseline)

ctl_mask = train.cp_type=='ctl_vehicle'

train = train[~ctl_mask]

targets = targets[~ctl_mask]



# drop id col

X = train.iloc[:,1:].to_numpy()

X_test = test.iloc[:,1:].to_numpy()

y = targets.iloc[:,1:].to_numpy() 
clf = Pipeline([('encode', CountEncoder(cols=[0, 2])),

                ('classify', MultiOutputClassifier(

                                 XGBClassifier(tree_method='gpu_hist')))

               ])
params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



clf.set_params(**params)
oof_preds = np.zeros(y.shape)

test_preds = np.zeros((test.shape[0], y.shape[1]))

kf = KFold(n_splits=NFOLDS)

for fn, (trn_idx, val_idx) in enumerate(kf.split(X, y)):

    print('Starting fold: ', fn)

    X_train, X_val = X[trn_idx], X[val_idx]

    y_train, y_val = y[trn_idx], y[val_idx]

    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val) # list of preds per class

    val_preds = np.array(val_preds)[:,:,1].T # take the positive class

    oof_preds[val_idx] = val_preds

    

    preds = clf.predict_proba(X_test)

    preds = np.array(preds)[:,:,1].T # take the positive class

    test_preds += preds / NFOLDS
print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# setting control test preds to 0

control_mask = [test['cp_type']=='ctl_vehicle']



test_preds[control_mask] = 0



# createing submission file

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)