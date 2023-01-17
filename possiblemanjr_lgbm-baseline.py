import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score 

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve ,auc , log_loss ,  classification_report 

from sklearn.preprocessing import StandardScaler , Binarizer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

import time

import os, sys, gc, warnings, random, datetime

import math

import shap

import joblib

warnings.filterwarnings('ignore')



import xgboost as xgb

from sklearn.model_selection import StratifiedKFold , cross_val_score , KFold

from sklearn.metrics import roc_auc_score
train = pd.read_pickle('../input/basic-eda-with-visualization-data-preprocessing/train.pkl')

test = pd.read_pickle('../input/basic-eda-with-visualization-data-preprocessing/test.pkl')
train.head()
remove_features = ['id' , 'income']
def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
# VARS



SEED = 2020

seed_everything(SEED)

target = 'income'
###### Model params

lgb_params = {

                    'objective':'binary',

                    'boosting_type':'gbdt',

                    'metric':'auc',

                    'n_jobs':-1,

                    'learning_rate':0.005,

                    'num_leaves': 2**7,

                    'max_depth':-1,

                    'tree_learner':'serial',

                    'colsample_bytree': 0.8,

                    'subsample_freq':1,

                    'subsample':0.8,

                    'n_estimators':100000,

                    'max_bin':127,

                    'verbose':-1,

                    'seed': SEED,

                    'early_stopping_rounds':100, 

                } 
from sklearn.metrics import f1_score    

def lgb_f1_score(y_hat,data):

    y_true = data.get_label()

    y_hat = np.round(y_hat) 

    return 'f1', f1_score(y_true, y_hat,average='macro'), True
features_columns = [col for col in list(train) if col not in remove_features]



folds = KFold(n_splits=5, shuffle=True, random_state=SEED)



X,y = train[features_columns], train[target]    

P = test[features_columns] 



    

predictions = np.zeros(len(test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):

    print('Fold:',fold_)

    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]

    vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]



    print(len(tr_x),len(vl_x))

    tr_data = lgb.Dataset(tr_x, label=tr_y)

    vl_data = lgb.Dataset(vl_x, label=vl_y)  



    estimator = lgb.train(

        lgb_params,

        tr_data,

        feval= lgb_f1_score, 

        valid_sets = [tr_data, vl_data],

        verbose_eval = 100

    )   



    pp_p = estimator.predict(P)

    predictions += pp_p/5





test['prediction'] = predictions



submission = test[['id','prediction']] 

submission['prediction'] = submission['prediction'].round(0)

submission['prediction'] = submission['prediction'].astype(int)
submission.to_csv('sub_lgbm_5kfold_simple_f1', index = False)