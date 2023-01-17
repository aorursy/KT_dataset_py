# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import gc

import time

from scipy import interp



from contextlib import contextmanager

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier



from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score,precision_recall_curve

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

import numpy as np

import gc

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import imblearn

from mpl_toolkits.mplot3d import Axes3D

np.random.seed(5)

from sklearn import decomposition, datasets 

from imblearn.over_sampling import SMOTE

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc

from sklearn import metrics



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import cross_validate

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler



from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
train_df = pd.read_csv('../input/preprocessed-home-credit/local_data_final.csv')

test_df = pd.read_csv('../input/preprocessed-home-credit/kaggle_test_final.csv')

t = train_df.head(70000)
def normalize(t, test_df):

    train_dff = t.drop(columns = ['SK_ID_CURR',"TARGET"],axis = 1)

    X_train = train_dff.values

    print (X_train.shape)

    Y_train = t['TARGET']

    

    test_dff = test_df.drop(columns = ['SK_ID_CURR',"TARGET"],axis = 1)

    X_test = test_dff.values

    print (X_test.shape)



    scaler = preprocessing.StandardScaler().fit(X_train)

    print (scaler)

    xtr = pd.DataFrame((scaler.transform(X_train)).tolist())

#     xtr['TARGET'] = Y_train.values.tolist()



    xte = pd.DataFrame((scaler.transform(X_test)).tolist())



    # df_norm = shuffle(df_norm.values)

    # df_norm = pd.DataFrame(df_norm.tolist())

    return xtr, xte, Y_train
X_train, X_test, Y_train = normalize (t, test_df)
dff = train_df.drop('SK_ID_CURR', axis = 1)

dfff = dff.drop('TARGET', axis = 1)

target = dff['TARGET']

X_train.columns = dfff.columns

X_test.columns = dfff.columns

train_dff = X_train

train_dff['TARGET'] = Y_train.tolist()
test_dff = X_test

test_dff.head()
def learning_rate_010_decay_power_099(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_010_decay_power_0995(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.995, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_005_decay_power_099(current_iter):

    base_learning_rate = 0.05

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3



import lightgbm as lgb

fit_params={"early_stopping_rounds":30, 

            "eval_metric" : 'auc', 

#             "eval_set" : [(X_test,Y_test)],

            'eval_names': ['valid'],

            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

            'verbose': 100,

            'categorical_feature': 'auto'}



from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}





#This parameter defines the number of HP points to be tested

n_HP_points_to_test = 100



import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)

gs = RandomizedSearchCV(

    estimator=clf, param_distributions=param_test, 

    n_iter=n_HP_points_to_test,

    scoring='roc_auc',

    cv=3,

    refit=True,

    random_state=314,

    verbose=True)





opt_parameters = {'colsample_bytree': 0.48263575356020577, 'min_child_samples': 311, 

                  'min_child_weight': 1, 'num_leaves': 7, 'reg_alpha': 7, 'reg_lambda': 0.1, 

                  'subsample': 0.3542761367404292} 





# LightGBM GBDT with KFold or Stratified KFold

def kfold_lightgbm(train_df,test_df,Y_train,num_folds,stratified = False, debug= False):

    # Cross validation model

    if stratified:

        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)

    else:

        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)

    # Create arrays and dataframes to store results

    oof_preds = np.zeros(train_df.shape[0])

    sub_preds = np.zeros(test_df.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):

        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]

        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]



        # LightGBM parameters found by Bayesian optimization

        clf = LGBMClassifier(boosting_type='gbdt', class_weight=None,

        colsample_bytree=0.48263575356020577, importance_type='split',

        learning_rate=0.1, max_depth=-1, metric='None',

        min_child_samples=311, min_child_weight=1, min_split_gain=0.0,

        n_estimators=10000, n_jobs=4, num_leaves=7, objective=None,

        random_state=None, reg_alpha=7, reg_lambda=0.1, silent=True,

        subsample=0.3542761367404292, subsample_for_bin=200000,

        subsample_freq=0,scale_pos_weight = 2)





        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 30,callbacks=[lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_0995)])



        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        

        fold_importance_df = pd.DataFrame()

        fold_importance_df["feature"] = feats

        fold_importance_df["importance"] = clf.feature_importances_

        fold_importance_df["fold"] = n_fold + 1

        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

        

        del clf, train_x, train_y, valid_x, valid_y

        gc.collect()

        

#     print (sub_preds)

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    

    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(train_df[feats], train_df['TARGET'])]

    display_roc_curve(y_=Y_train,oof_preds_=oof_preds,sub_preds_ = sub_preds, folds_idx_=folds_idx)

    display_precision_recall(y_=Y_train, oof_preds_=oof_preds, folds_idx_=folds_idx)

    

    return feature_importance_df,sub_preds



def display_roc_curve(y_, oof_preds_,sub_preds_,folds_idx_):

    # Plot ROC curves

    plt.figure(figsize=(6,6))

    scores = [] 

    for n_fold, (_, val_idx) in enumerate(folds_idx_):  

        # Plot the roc curve

        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])

        score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])

        scores.append(score)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)

    score = roc_auc_score(y_, oof_preds_)

    plt.plot(fpr, tpr, color='b',

             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),

             lw=2, alpha=.8)

    

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('LightGBM ROC Curve')

    plt.legend(loc="lower right")

    plt.tight_layout()

    

    plt.savefig('roc_curve.png')



def display_precision_recall(y_, oof_preds_, folds_idx_):

    # Plot ROC curves

    plt.figure(figsize=(6,6))

    

    scores = [] 

    for n_fold, (_, val_idx) in enumerate(folds_idx_):  

        # Plot the roc curve

        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])

        score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])

        scores.append(score)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    

    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)

    score = average_precision_score(y_, oof_preds_)

    plt.plot(precision, recall, color='b',

             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),

             lw=2, alpha=.8)

    

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.title('LightGBM Recall / Precision')

    plt.legend(loc="best")

    plt.tight_layout()

    

    

def compiler2(train_df,test_df,Y_train,debug = False):

    num_rows = 10000 if debug else None

    feat_importance,sub_preds = kfold_lightgbm(train_df,test_df,

                                                Y_train, num_folds= 5, 

                                                stratified= True, debug= debug)

    

    return feat_importance,sub_preds

feat_importance,sub_preds = compiler2(train_dff,test_dff,Y_train,debug = False)

result = pd.DataFrame()

result['SK_ID_CURR'] = test_df['SK_ID_CURR'].values.tolist()

result['TARGET'] = sub_preds

result.to_csv('submission_kernel_lightGBM.csv', index= False)

# coding: utf-8

# pylint: disable = invalid-name, C0111

import lightgbm as lgb

import pandas as pd



if lgb.compat.MATPLOTLIB_INSTALLED:

    import matplotlib.pyplot as plt

else:

    raise ImportError('You need to install matplotlib for plot_example.py.')



# create dataset for lightgbm

# X_train = X_train.drop('TARGET', axis = 1)

X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X_train, Y_train, test_size=0.1,random_state=42)

lgb_train = lgb.Dataset(X_train_, Y_train_)

lgb_test = lgb.Dataset(X_test_, Y_test_, reference=lgb_train)



# specify your configurations as a dict

params = {'colsample_bytree': 0.48263575356020577, 'min_child_samples': 311, 

                  'min_child_weight': 1, 'num_leaves': 7, 'reg_alpha': 7, 'reg_lambda': 0.1, 

                  'subsample': 0.3542761367404292,  'metric': 'roc_auc'} 





print('Starting training...')

# train

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=100,

                valid_sets=[lgb_train, lgb_test],

                feature_name=[X_train.columns.tolist()[i] for i in range(len(X_train.columns.tolist()))],

                verbose_eval=10)



_ = lgb.plot_tree(gbm, figsize=(30, 50))
