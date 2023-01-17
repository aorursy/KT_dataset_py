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

warnings.simplefilter('ignore')
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

sub = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')
train[train['sig_id'].isin(list(test['sig_id'].values))]
SEED = 2020

split = 10



np.random.seed(SEED)
# drop id col

X = train.iloc[:,1:]

X_test = test.iloc[:,1:]

y = targets.iloc[:,1:]
MODEL = Pipeline([('encode', CountEncoder(cols=['cp_type', 'cp_dose'])),

                ('classify', MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist' ,

                                                                 colsample_bytree=0.3,

                                                                 seed=1000,gpu_id = 0,)))

                       

               ])
#group Kfold

from sklearn.model_selection import GroupKFold



grk = GroupKFold(n_splits=5)

groups = train.sig_id

sub_preds = np.zeros((test.shape[0], y.shape[1]))

oof_preds = np.zeros(y.shape)



#groups is the customer_id columns ,

for i,(tr_index,ts_index) in enumerate(grk.split(X,groups=groups)):

    print(f"FOLD {i+1}/{grk.n_splits}")

    X_train ,X_val = X.iloc[tr_index] , X.iloc[ts_index]

    y_train ,y_val = y.iloc[tr_index] , y.iloc[ts_index]

    

    # Model

    print('Fit ...')

    MODEL.fit(X_train, y_train,)

    

    print('Predict On validation ...')

    val_preds = MODEL.predict_proba(X_val)

    val_preds = np.array(val_preds)[:,:,1].T # take the positive class prediction

    oof_preds[ts_index] = val_preds

    # TEST PREDS

    print('Predict On Test ...')

    preds = MODEL.predict_proba(X_test ) 

    preds = np.array(preds)[:,:,1].T



    sub_preds += preds / 5                                  # grk.n_splits
print('OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))
# create the submission file

sub.iloc[:,1:] = sub_preds

sub.to_csv('submission.csv', index=False)