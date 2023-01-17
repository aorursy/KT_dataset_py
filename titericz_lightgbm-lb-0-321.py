import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import librosa

import matplotlib.pyplot as plt

import gc



from tqdm import tqdm, tqdm_notebook

from sklearn.metrics import label_ranking_average_precision_score

from sklearn.metrics import roc_auc_score



from joblib import Parallel, delayed

import lightgbm as lgb

from scipy import stats



from sklearn.model_selection import KFold



import warnings

warnings.filterwarnings('ignore')



def calculate_overall_lwlrap_sklearn(truth, scores):

    """Calculate the overall lwlrap using sklearn.metrics.lrap."""

    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.

    sample_weight = np.sum(truth > 0, axis=1)

    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

    overall_lwlrap = label_ranking_average_precision_score(

        truth[nonzero_weight_sample_indices, :] > 0, 

        scores[nonzero_weight_sample_indices, :], 

        sample_weight=sample_weight[nonzero_weight_sample_indices])

    return overall_lwlrap



tqdm.pandas()
test = pd.read_csv('../input/freesound-audio-tagging-2019/sample_submission.csv')



label_columns = list( test.columns[1:] )

label_mapping = dict((label, index) for index, label in enumerate(label_columns))



print(test.shape)
X     = np.load( '../input/freesoundpreproc1/LGB-train-1.npy' )

Xtest = np.load( '../input/freesoundpreproc1/LGB-test-1.npy' )

Y     = np.load( '../input/freesoundpreproc1/LGB-target.npy' )



X.shape, Xtest.shape, Y.shape
n_fold = 10

folds = KFold(n_splits=n_fold, shuffle=True, random_state=69)



params = {'num_leaves': 15,

         'min_data_in_leaf': 200, 

         'objective':'binary',

         "metric": 'auc',

         'max_depth': -1,

         'learning_rate': 0.05,

         "boosting": "gbdt",

         "bagging_fraction": 0.85,

         "bagging_freq": 1,

         "feature_fraction": 0.20,

         "bagging_seed": 42,

         "verbosity": -1,

         "nthread": -1,

         "random_state": 69}



PREDTRAIN = np.zeros( (X.shape[0],80) )

PREDTEST  = np.zeros( (Xtest.shape[0],80) )

for f in range(len(label_columns)):

    y = Y[:,f] #target label

    oof      = np.zeros( X.shape[0] )

    oof_test = np.zeros( Xtest.shape[0] )

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X,y)):

        model = lgb.LGBMClassifier(**params, n_estimators = 20000)

        model.fit(X[trn_idx,:], 

                  y[trn_idx], 

                  eval_set=[(X[val_idx,:], y[val_idx])], 

                  eval_metric='auc',

                  verbose=0, 

                  early_stopping_rounds=25)

        oof[val_idx] = model.predict_proba(X[val_idx,:], num_iteration=model.best_iteration_)[:,1]

        oof_test    += model.predict_proba(Xtest       , num_iteration=model.best_iteration_)[:,1]/n_fold



    PREDTRAIN[:,f] = oof    

    PREDTEST [:,f] = oof_test

    

    print( f, str(roc_auc_score( y, oof ))[:6], label_columns[f] )

    

print( 'Competition Metric Lwlrap cv:', calculate_overall_lwlrap_sklearn( Y, PREDTRAIN ) )
test[label_columns] = PREDTEST

test.to_csv('submission.csv', index=False)

test.head()