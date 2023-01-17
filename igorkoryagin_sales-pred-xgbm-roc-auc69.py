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

from xgboost import XGBClassifier
import xgboost as xgb
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
param_file = 'XGB_69auc_results'
df = load_saved_dataset(filename)
parameters = load_saved_parameters(param_file)
y = df["buy"]
df.drop(['buy'], 1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=1, test_size=0.2, shuffle=False)
del df

gc.collect()
results = {}
def run_xgb(results, X_train, X_test, y_train, y_test, parameters = None):
    
   
    tr_data = xgb.DMatrix(X_train, y_train)
    va_data = xgb.DMatrix(X_test, y_test)
    
    train_labels = tr_data.get_label()
    ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
    
        
    params = {'objective': 'binary:logistic',
              'booster': 'gbtree',
              'lambda': 1.833646534241334e-06, 'alpha': 1.9473014117037142e-07,
        'eval_metric': 'auc',
        'eta': 0.005,
        'max_depth': 9,
        'gamma': 2.0564949063472374e-08,
        'subsample': 0.7, 
        'colsample_bytree': 0.7,
       # 'random_state': 42, 
        'silent': True,
        #'n_estimators':90,
            'grow_policy': 'lossguide',
             'scale_pos_weight':ratio}
    
    if parameters is not None:
        params.update(parameters)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 1000, watchlist, maximize=False, early_stopping_rounds = 30,\
                          verbose_eval=10)
    
    dtest = xgb.DMatrix(X_test)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    preds = model_xgb.predict(va_data)
    #pred_labels = np.rint(preds)
    pred_labels = np.where(preds > 0,1,0)
    
    
    accuracy = accuracy_score(y_test, pred_labels)
    roc_auc = roc_auc_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels)
    rec = recall_score(y_test, pred_labels)
    
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
                    'recall':rec
                   })
    
    return model_xgb, results, pred_labels, preds
xgb_model, results, y_pred, y_pred_proba  = run_xgb(results, X_train, X_test, y_train, y_test, parameters['parameters'])
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='XGBoost')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'XGBoost ROC curve')
plt.show()