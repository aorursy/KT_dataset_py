# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.tabular.transform import add_cyclic_datepart
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Metrics for models evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import pickle # for saving and loading processed datasets and hyperparameters
import gc

import lightgbm as lgb
from xgboost import XGBClassifier
import optuna # for hyperparameter tuning

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
# Any results you write to the current directory are saved as output.
PATH = '/kaggle/input/recsys-challenge-2015'
def load_saved_dataset(filename):
    try:
        with open('../input/recsys-preprocessed/{}.pickle'.format(filename), 'rb') as fin:
            X = pickle.load(fin)
        print('Dataset loaded')
    except FileNotFoundError:
        print('File with saved dataset not found')
    return X

def load_saved_parameters(filename):
    try:
        with open('../input/recsys-parameters/{}.pickle'.format(filename), 'rb') as fin:
            X = pickle.load(fin)
        print('Parameters loaded')
    except FileNotFoundError:
        print('File with saved parameters not found')
    return X
filename = 'Processed_recsys'
param_file = 'LGBM_results_auc70'
df = load_saved_dataset(filename)
parameters = load_saved_parameters(param_file)
y = df["buy"]
df.drop(['buy'], 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1, test_size=0.2, shuffle=False)
del df

gc.collect()
# Dictionary for collecting results
results = {}
def run_lgbm(X_train, X_test, y_train, y_test, results, parameters = None):

    print("Building datasets for lightgbm")
    
    dtrain = lgb.Dataset(X_train,label=y_train,feature_name=X_train.columns.tolist())
    dvalid = lgb.Dataset(X_test,label=y_test,feature_name=X_train.columns.tolist())
    
    train_labels = dtrain.get_label()

    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)

    
    params = {
        'boosting_type': 'gbdt', 
        'objective': 'binary', 'metric': 'auc',
        'learning_rate': 0.03, 'num_leaves': 10, 'max_depth': 12,
        'min_child_samples':100, # min_data_in_leaf
        'max_bin': 50, #number of bucked bin for feature values
        'subsample': 0.9, # subsample ratio of the training instance
        'subsample_freq':1, # frequence of subsample
        'colsample_bytree': 0.9, # subsample ratio of columns when constructing each tree.
        'min_child_weight':0,
        'min_split_gain':0, # lambda_l1, lambda_l2 and min_gain_to_split to regularization.
        'nthread':8, 'verbose': 0, 
        'scale_pos_weight': ratio, # because training data is extremely unbalanced
        'lambda_l1': 8.83655320341901e-06, 'lambda_l2': 9.47268218843624
    }
    
      
    if parameters is not None:
        params.update(parameters)
        
    evals_results = {}
    
    print("Starting classification")
    
    model = lgb.train(params, dtrain, valid_sets=[dtrain,dvalid],
                      #categorical_feature=cats,
                      valid_names=['train','test'],
                      evals_result = evals_results, num_boost_round=400,
                      early_stopping_rounds=40,
                      verbose_eval=20, feval = None)
    
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    #y_pred = np.round_(y_pred_prob, 0)
    y_pred = np.where(y_pred_proba > 0.1,1,0)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    
    print('The accuracy of prediction is:', accuracy)
    print('The ROC AUC of prediction is:', roc_auc)
    print('The F1 Score of prediction is:', f1)
    print('The Precision of prediction is:', prec)
    print('The Recall of prediction is:', rec)
    
    results.update({'parameters':params,
                   'accuracy':accuracy,
                    'roc_auc':roc_auc,
                     'f1':f1,
                    'precision':prec,
                    'recall':rec,
                    'evals_results':evals_results
                   })
    
    
    return model, results, y_pred, y_pred_proba
model, results, y_pred, y_pred_proba = run_lgbm(X_train, X_test, y_train, y_test,results, parameters['parameters'])
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='LGBM')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'LGBM ROC curve')
plt.show()