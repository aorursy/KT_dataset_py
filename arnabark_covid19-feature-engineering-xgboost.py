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
#!pip install seaborn

# !pip install --upgrade catboost

import catboost

import optuna

import imblearn

from catboost import CatBoostRegressor

from imblearn.under_sampling import RandomUnderSampler

import numpy as np

import pandas as pd

from catboost import *

import matplotlib.pyplot as plt

import seaborn as sns

from catboost import Pool

from datetime import datetime

from numpy import mean

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.linear_model import LinearRegression,RidgeCV

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

from scipy.stats import norm,skew

from scipy import stats

from sklearn.metrics import mean_squared_error,make_scorer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

import pandas as pd

import nltk

import operator

import re

import sys

from scipy import stats

from nltk.corpus import stopwords

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

# from multiprocessing import Pool

nltk.download("stopwords")

nltk.download("punkt")

import statsmodels.api as sm

from statsmodels.formula.api import ols

import time
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')

train.shape, test.shape, ss.shape
train.columns
test.head()
# S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
ss.columns
test['E']=[sum([i=='E' for i in j])/len(j) for j in test['predicted_loop_type']]

test['S']=[sum([i=='S' for i in j])/len(j) for j in test['predicted_loop_type']]

test['B']=[sum([i=='B' for i in j])/len(j) for j in test['predicted_loop_type']]

test['H']=[sum([i=='H' for i in j])/len(j) for j in test['predicted_loop_type']]

test['I']=[sum([i=='I' for i in j])/len(j) for j in test['predicted_loop_type']]

test['X']=[sum([i=='X' for i in j])/len(j) for j in test['predicted_loop_type']]

test['M']=[sum([i=='M' for i in j])/len(j) for j in test['predicted_loop_type']]



test['G']=[sum([i=='G' for i in j])/len(j) for j in test['sequence']]

test['A']=[sum([i=='A' for i in j])/len(j) for j in test['sequence']]

test['C']=[sum([i=='C' for i in j])/len(j) for j in test['sequence']]

test['U']=[sum([i=='U' for i in j])/len(j) for j in test['sequence']]

test['Paired']=[sum([i=='(' or i==')' for i in j])/len(j) for j in test['structure']]

test['Unpaired']=[sum([i=='.' for i in j])/len(j) for j in test['structure']]

train['E']=[sum([i=='E' for i in j])/len(j) for j in train['predicted_loop_type']]

train['S']=[sum([i=='S' for i in j])/len(j) for j in train['predicted_loop_type']]

train['B']=[sum([i=='B' for i in j])/len(j) for j in train['predicted_loop_type']]

train['H']=[sum([i=='H' for i in j])/len(j) for j in train['predicted_loop_type']]

train['I']=[sum([i=='I' for i in j])/len(j) for j in train['predicted_loop_type']]

train['X']=[sum([i=='X' for i in j])/len(j) for j in train['predicted_loop_type']]

train['M']=[sum([i=='M' for i in j])/len(j) for j in train['predicted_loop_type']]

train['G']=[sum([i=='G' for i in j])/len(j) for j in train['sequence']]

train['A']=[sum([i=='A' for i in j])/len(j) for j in train['sequence']]

train['C']=[sum([i=='C' for i in j])/len(j) for j in train['sequence']]

train['U']=[sum([i=='U' for i in j])/len(j) for j in train['sequence']]

train['Paired']=[sum([i=='(' or i==')' for i in j])/len(j) for j in train['structure']]

train['Unpaired']=[sum([i=='.' for i in j])/len(j) for j in train['structure']]
train.columns
# train['reactivity']=[np.mean(x) for x in train['reactivity']]

# train['deg_error_Mg_pH10']=[np.mean(x) for x in train['deg_Mg_pH10']]

# train['deg_error_pH10']=[np.mean(x) for x in train['deg_pH10']]

# train['deg_error_Mg_50C']=[np.mean(x) for x in train['deg_Mg_50C']]

# train['deg_error_50C']=[np.mean(x) for x in train['deg_50C']]

# train=train[['id', 'sequence', 'structure', 'predicted_loop_type',

#        'signal_to_noise', 'SN_filter', 'seq_length', 'seq_scored',

#        'reactivity_error', 'deg_error_Mg_pH10', 'deg_error_pH10',

#        'deg_error_Mg_50C', 'deg_error_50C', 'reactivity', 'deg_Mg_pH10',

#        'deg_pH10', 'deg_Mg_50C', 'deg_50C', 'G', 'A', 'C', 'U', 'Paired',

#        'Unpaired']]
for a in [ 'G', 'A', 'C', 'U']:

    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]
for a in [ 'E', 'S', 'H',]:

    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]
for a in [ 'E', 'S', 'H',]:

    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]

    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]
a='S'

[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
# import seaborn as sns

# plt.subplots(figsize=(20,10))

# sns.heatmap(train.corr()[[ 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C',

#        'deg_50C']],annot=True)
train_ex=pd.DataFrame()

for index in train.index:

    temp=pd.DataFrame()

    temp['id_seqpos']=[str(str(train['id'][index])+'_'+str(i)) for i in range(train['seq_scored'][index])]

#     temp['sequence']=[train['sequence'][index][i] for i in range(train['seq_scored'][index])]

    temp['sequence_loop']=[str(train['sequence'][index][i]+train['predicted_loop_type'][index][i]) for i in range(train['seq_scored'][index])]

    temp['structure']=[train['structure'][index][i] for i in range(train['seq_scored'][index])]

#     temp['predicted_loop_type']=[train['predicted_loop_type'][index][i] for i in range(train['seq_scored'][index])]

    for r in range(1,20):

        temp[str(str(r)+'forward_predicted_loop_type')]=[train['predicted_loop_type'][index][i+r] if i+r<train['seq_scored'][index] else -1 for i in range(train['seq_scored'][index])]

        temp[str(str(r)+'backward_predicted_loop_type')]=[train['predicted_loop_type'][index][i-r] for i in range(train['seq_scored'][index])]

        temp[str(str(r)+'forward_structure')]=[train['structure'][index][i+r] if i+r<train['seq_scored'][index] else -1 for i in range(train['seq_scored'][index])]

        temp[str(str(r)+'backward_structure')]=[train['structure'][index][i-r] for i in range(train['seq_scored'][index])]

        temp[str(str(r)+'forward_sequence')]=[train['sequence'][index][i+r] if i+r<train['seq_scored'][index] else -1 for i in range(train['seq_scored'][index])]

        temp[str(str(r)+'backward_sequence')]=[train['sequence'][index][i-r] for i in range(train['seq_scored'][index])]

    temp['E']=train['E'][index]

    temp['S']=train['S'][index]

    temp['B']=train['B'][index]

    temp['H']=train['H'][index]

    temp['I']=train['I'][index]

    temp['G']=train['G'][index]

    temp['A']=train['A'][index]

    temp['C']=train['C'][index]

    temp['U']=train['U'][index]

    temp['index']=[i for i in range(train['seq_scored'][index])]

    temp['Paired']=train['Paired'][index]

    temp['Unpaired']=train['Unpaired'][index]

    temp['G_position']=train['G_position'][index]

    temp['A_position']=train['A_position'][index]

    temp['C_position']=train['C_position'][index]

    temp['U_position']=train['U_position'][index]

    temp['E_position']=train['E_position'][index]

    temp['S_position']=train['S_position'][index]

    temp['H_position']=train['H_position'][index]

    temp['reactivity']=[train['reactivity'][index][i] for i in range(train['seq_scored'][index])]

    temp['deg_Mg_pH10']=[train['deg_Mg_pH10'][index][i] for i in range(train['seq_scored'][index])]

    temp['deg_pH10']=[train['deg_pH10'][index][i] for i in range(train['seq_scored'][index])]

    temp['deg_Mg_50C']=[train['deg_Mg_50C'][index][i] for i in range(train['seq_scored'][index])]

    temp['deg_50C']=[train['deg_50C'][index][i] for i in range(train['seq_scored'][index])]

    train_ex=train_ex.append(temp)



train_ex['sequence_loop'].unique()
test_ex=pd.DataFrame()

for index in test.index:

    temp=pd.DataFrame()

    temp['id_seqpos']=[str(str(test['id'][index])+'_'+str(i)) for i in range(test['seq_length'][index])]

#     temp['sequence']=[test['sequence'][index][i] for i in range(test['seq_length'][index])]

    temp['sequence_loop']=[str(test['sequence'][index][i]+test['predicted_loop_type'][index][i]) for i in range(test['seq_length'][index])]

    temp['structure']=[test['structure'][index][i] for i in range(test['seq_length'][index])]

#     temp['predicted_loop_type']=[test['predicted_loop_type'][index][i] for i in range(test['seq_length'][index])]

    

    for r in range(1,20):

        temp[str(str(r)+'forward_predicted_loop_type')]=[test['predicted_loop_type'][index][i+r] if i+r<test['seq_length'][index] else -1 for i in range(test['seq_length'][index])]

        temp[str(str(r)+'backward_predicted_loop_type')]=[test['predicted_loop_type'][index][i-r] for i in range(test['seq_length'][index])]

        temp[str(str(r)+'forward_structure')]=[test['structure'][index][i+r] if i+r<test['seq_length'][index] else -1 for i in range(test['seq_length'][index])]

        temp[str(str(r)+'backward_structure')]=[test['structure'][index][i-r] for i in range(test['seq_length'][index])]

        temp[str(str(r)+'forward_sequence')]=[test['sequence'][index][i+r] if i+r<test['seq_length'][index] else -1 for i in range(test['seq_length'][index])]

        temp[str(str(r)+'backward_sequence')]=[test['sequence'][index][i-r] for i in range(test['seq_length'][index])]

    temp['E']=test['E'][index]

    temp['S']=test['S'][index]

    temp['B']=test['B'][index]

    temp['H']=test['H'][index]

    temp['I']=test['I'][index]

    temp['G']=test['G'][index]

    temp['A']=test['A'][index]

    temp['C']=test['C'][index]

    temp['U']=test['U'][index]

    temp['index']=[i for i in range(test['seq_length'][index])]

    temp['Paired']=test['Paired'][index]

    temp['Unpaired']=test['Unpaired'][index]

    temp['G_position']=test['G_position'][index]

    temp['A_position']=test['A_position'][index]

    temp['C_position']=test['C_position'][index]

    temp['U_position']=test['U_position'][index]

    temp['E_position']=test['E_position'][index]

    temp['S_position']=test['S_position'][index]

    temp['H_position']=test['H_position'][index]

    test_ex=test_ex.append(temp)



result=test_ex
test_columns=[i for i in np.intersect1d(test_ex.columns,train_ex.columns) if i!='id_seqpos']

x_test=test_ex[test_columns]

x_t=train_ex[test_columns]

x_test=pd.get_dummies(x_test)

x_t=pd.get_dummies(x_t)

test_dum_columns=[i for i in np.intersect1d(x_t.columns,x_test.columns)]

y_t=train_ex[['reactivity', 'deg_Mg_pH10',

       'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

x_t=x_t[test_dum_columns]

x_test=x_test[test_dum_columns]
scaler = StandardScaler()

scaler.fit(x_t)



x_t = scaler.transform(x_t)

x_test = scaler.transform(x_test)
x_train,x_valid,y_train,y_valid=train_test_split(x_t,y_t,test_size=0.1,shuffle=True)

print(x_train.shape)

print(y_train.shape)
import optuna

import xgboost as xgb

import sklearn

column='reactivity'

def objective(trial):

    column='reactivity'

    dtrain = xgb.DMatrix(x_train, label=y_train[column])

    dvalid = xgb.DMatrix(x_valid, label=y_valid[column])

    

    param = {

        "silent": 1,

#           "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),

          "eval_metric": "rmse",

        "booster": "gbtree",

        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),

        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),

        'tree_method' : 'gpu_hist'

        

    }



    if param["booster"] == "gbtree" or param["booster"] == "dart":

        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

#         param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)

#         param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)

#         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])





    # Add a callback for pruning.

#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))

    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")])

    preds = bst.predict(dvalid)

    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))

    return rmse



study = optuna.create_study()

study.optimize(objective, n_trials=200)
print(study.best_params)

dtrain = xgb.DMatrix(x_train, label=y_train[column])

dvalid = xgb.DMatrix(x_valid, label=y_valid[column])

dtest = xgb.DMatrix(x_test)

bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])

preds = bst.predict(dtest)

result[column]=preds
import optuna

import xgboost as xgb

import sklearn

column='deg_Mg_pH10'

def objective(trial):

    column='deg_Mg_pH10'

    dtrain = xgb.DMatrix(x_train, label=y_train[column])

    dvalid = xgb.DMatrix(x_valid, label=y_valid[column])



    param = {

        "silent": 1,

#           "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),

          "eval_metric": "rmse",

        "booster": "gbtree",

        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),

        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),

        'tree_method' : 'gpu_hist'

        

    }



    if param["booster"] == "gbtree" or param["booster"] == "dart":

        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

#         param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)

#         param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)

#         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])





    # Add a callback for pruning.

#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))

    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")])

    preds = bst.predict(dvalid)

    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))

    return rmse

study = optuna.create_study()

study.optimize(objective, n_trials=200)

study.best_params
dtrain = xgb.DMatrix(x_train, label=y_train[column])

dvalid = xgb.DMatrix(x_valid, label=y_valid[column])

dtest = xgb.DMatrix(x_test)

bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])

preds = bst.predict(dtest)

result[column]=preds
import optuna

import xgboost as xgb

import sklearn

column='deg_Mg_50C'

def objective(trial):

    column='deg_Mg_50C'

    dtrain = xgb.DMatrix(x_train, label=y_train[column])

    dvalid = xgb.DMatrix(x_valid, label=y_valid[column])



    param = {

        "silent": 1,

#           "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),

          "eval_metric": "rmse",

        "booster": "gbtree",

        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),

        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),

        'tree_method' : 'gpu_hist'

        

    }



    if param["booster"] == "gbtree" or param["booster"] == "dart":

        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

#         param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)

#         param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)

#         param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])





    # Add a callback for pruning.

#     pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))

    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")])

    preds = bst.predict(dvalid)

    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))

    return rmse

study = optuna.create_study()

study.optimize(objective, n_trials=200)

study.best_params
dtrain = xgb.DMatrix(x_train, label=y_train[column])

dvalid = xgb.DMatrix(x_valid, label=y_valid[column])

dtest = xgb.DMatrix(x_test)

bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])

preds = bst.predict(dtest)

result[column]=preds
result.head()
result['deg_pH10']=0

result['deg_50C']=0

result[['id_seqpos','reactivity', 'deg_Mg_pH10',

       'deg_pH10', 'deg_Mg_50C', 'deg_50C']].head()
result[['id_seqpos','reactivity', 'deg_Mg_pH10',

       'deg_pH10', 'deg_Mg_50C', 'deg_50C']].to_csv('submission.csv',index=False)