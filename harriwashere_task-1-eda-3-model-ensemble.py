# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from sklearn.datasets import load_breast_cancer
!pip install optuna


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import lightgbm as lgb
import xgboost as xgb
# sklearn
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

import sklearn.metrics

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import operator as op 
from itertools import combinations

from sklearn.metrics  import accuracy_score, auc, roc_curve, precision_recall_curve, roc_auc_score, precision_score, recall_score, average_precision_score

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from itertools import combinations, permutations
# Display all columns
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-colorblind')

df = pd.read_excel('/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')


# source: https://www.kaggle.com/felipeveiga/starter-covid-19-sirio-libanes-icu-admission
comorb_lst = [i for i in df.columns if "DISEASE" in i]
comorb_lst.extend(["HTN", "IMMUNOCOMPROMISED", "OTHER"])

demo_lst = [i for i in df.columns if "AGE_" in i]
demo_lst.append("GENDER")

vitalSigns_lst = df.iloc[:,193:-2].columns.tolist()

lab_lst = df.iloc[:,13:193].columns.tolist()
### Redundant Feature Check
def is_one_to_one(df, cols):
    """Check whether any number of columns are one-to-one match.

    df: a pandas.DataFrame
    cols: must be a list of columns names

    Duplicated matches are allowed:
        a - 1
        b - 2
        b - 2
        c - 3
    (This two cols will return True)
    Source: [link](https://stackoverflow.com/questions/50643386/easy-way-to-see-if-two-columns-are-one-to-one-in-pandas)

    """
    if len(cols) == 1:
        return True
        # You can define you own rules for 1 column check, Or forbid it

    # MAIN THINGs: for 2 or more columns check!
    res = df.groupby(cols).count()
    uniqueness = [res.index.get_level_values(i).is_unique
                for i in range(res.index.nlevels)]
    return all(uniqueness)

# Getting combinations of all the colmns
combos = list(combinations(df.columns,2))

# Running to see if any of them are identical
identical_cols = []

for col in np.arange(0,len(combos),1):
    x = [combos[col][0],combos[col][1]]
    if is_one_to_one(df,x) == True:
         identical_cols.append(combos[col][0])
all_cols = [x for x in df.columns if x not in identical_cols]
df = df.loc[:, all_cols]
df.info()
# source cell 6: https://www.kaggle.com/fernandoramacciotti/interpretable-icu-risk-0-2-window-only
# missing values
df = df\
    .sort_values(by=['PATIENT_VISIT_IDENTIFIER', 'WINDOW'])\
    .groupby('PATIENT_VISIT_IDENTIFIER', as_index=False)\
    .fillna(method='ffill')\
    .fillna(method='bfill')
df = df.set_index('PATIENT_VISIT_IDENTIFIER') 
w02 = df[df.WINDOW == '0-2']
w24 = df[df.WINDOW == '2-4']
w46 = df[df.WINDOW == '4-6']
w612 = df[df.WINDOW == '6-12']
wa12 = df[df.WINDOW == 'ABOVE_12']

w02['ICU_W24'] = w24['ICU']
w02['ICU_W46'] = w46['ICU']
w02['ICU_W612'] = w612['ICU']
w02['ICU_Wa12'] = wa12['ICU'] 

w24['ICU_W46'] = w46['ICU']
w24['ICU_W612'] = w612['ICU']
w24['ICU_Wa12'] = wa12['ICU'] 

w46['ICU_W612'] = w612['ICU']
w46['ICU_Wa12'] = wa12['ICU'] 

w612['ICU_Wa12'] = wa12['ICU'] 
# FIRST REMOVE ICU 1 FROM WINDOW 0-2
w02 = w02[w02.ICU == 0]

# NEW TARGET "NOT ICU"
w02['temp'] = w02.loc[:,['ICU','ICU_W24','ICU_W46','ICU_W612','ICU_Wa12']].sum(axis=1) 
def label_icu(x):
    if (x['temp'] == 0):
        val = 0
    elif (x['temp'] > 0):
        val = 1
    return val

w02['EVENTUAL_ICU'] = w02.apply(label_icu, axis=1)

# REMOVE UNWANTED COLUMNS 
w02_df = w02.drop(['EVENTUAL_ICU', 'temp','WINDOW','ICU','ICU_W24','ICU_W46','ICU_W612','ICU_Wa12'], axis = 1)


# FIRST REMOVE ICU 1 FROM WINDOW 2-4
w24 = w24[w24.ICU == 0]

# NEW TARGET "NOT ICU"
w24['temp'] = w24.loc[:,['ICU','ICU_W46','ICU_W612','ICU_Wa12']].sum(axis=1) 
def label_icu(x):
    if (x['temp'] == 0):
        val = 0
    elif (x['temp'] > 0):
        val = 1
    return val

w24['EVENTUAL_ICU'] = w24.apply(label_icu, axis=1)

# REMOVE UNWANTED COLUMNS 
w24_df = w24.drop(['EVENTUAL_ICU', 'temp','WINDOW','ICU','ICU_W46','ICU_W612','ICU_Wa12'], axis = 1)

# FIRST REMOVE ICU 1 FROM WINDOW 4-6
w46 = w46[w46.ICU == 0]

# NEW TARGET "NOT ICU"
w46['temp'] = w46.loc[:,['ICU','ICU_W612','ICU_Wa12']].sum(axis=1) 
def label_icu(x):
    if (x['temp'] == 0):
        val = 0
    elif (x['temp'] > 0):
        val = 1
    return val

w46['EVENTUAL_ICU'] = w46.apply(label_icu, axis=1)

# REMOVE UNWANTED COLUMNS 
w46_df = w46.drop(['EVENTUAL_ICU', 'temp','WINDOW','ICU','ICU_W612','ICU_Wa12'], axis = 1)


# FIRST REMOVE ICU 1 FROM WINDOW 6-12
w612 = w612[w612.ICU == 0]

# NEW TARGET "NOT ICU"
w612['temp'] = w612.loc[:,['ICU','ICU_Wa12']].sum(axis=1) 
def label_icu(x):
    if (x['temp'] == 0):
        val = 0
    elif (x['temp'] > 0):
        val = 1
    return val

w612['EVENTUAL_ICU'] = w612.apply(label_icu, axis=1)

# REMOVE UNWANTED COLUMNS 
w612_df = w612.drop(['EVENTUAL_ICU', 'temp','WINDOW','ICU','ICU_Wa12'], axis = 1)
w02['EVENTUAL_ICU'].value_counts()
datacopy = w02_df
x =[]
for col in w02_df.columns:
  n = len(datacopy[col].unique())
  if (15 < n):
       continue
  if (15 > n > 2 ): # making a list of columns that are greater than 2 levels
      y = col
      x.append(y)  

w02X = pd.get_dummies(datacopy, columns = x)

X = w02X
y =  w02['EVENTUAL_ICU']
models = ['gp', 'et', 'xgb', 'gbm'] # when you want to try all of them/ iteration: 1 
comb = list(combinations(models, 3))


def scaler_fuc(scaler, X, y):

  train_x, valid_x, train_y, valid_y = train_test_split(X, y,
                                              test_size=0.25, random_state = 123)
  if (scaler == 'minmax'):
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    scaler.fit(valid_x)
    valid_x = scaler.transform(valid_x)

  if (scaler == 'stand'):
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    scaler.fit(valid_x)
    valid_x = scaler.transform(valid_x)

  if (scaler == 'log'):
    transformer = FunctionTransformer(np.log1p, validate=True)
    train_x = transformer.transform(train_x)
    valid_x = transformer.transform(valid_x)
    # prevent log 0 error
    np.seterr(divide = 'ignore')  # invalid value encountered in log1p
    train_x = np.where(np.isneginf(train_x), 0, train_x)
    valid_x = np.where(np.isneginf(valid_x), 0, valid_x)
    train_x = np.where(np.isinf(train_x), 0, train_x)
    valid_x = np.where(np.isinf(valid_x), 0, valid_x)
    train_x = np.where(np.isnan(train_x), 0, train_x)
    valid_x = np.where(np.isnan(valid_x), 0, valid_x)

  # turning these train/tests into lgb/xgb datasets
  dtrain_gbm = lgb.Dataset(train_x, label=train_y)
  dvalid_gbm = lgb.Dataset(valid_x, label=valid_y)

  dtrain_xbg = xgb.DMatrix(train_x, label=train_y)
  dvalid_xbg = xgb.DMatrix(valid_x, label=valid_y)
  return train_x, valid_x, train_y, valid_y,  dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg


class Objective:

    def __init__(self):
        self.best_gbm = None
        self._gbm = None
        self.best_xgb = None
        self._xgb = None
        self.predictions = None
        self.fpredictions = None

    def __call__(self, trial):

        i = trial.suggest_int("combos", 0, (len(comb)-1))
        gbm_preds = np.zeros((math.ceil(len(y)*0.25),1), dtype=np.int)
        xgb_preds = np.zeros((math.ceil(len(y)*0.25),1), dtype=np.int)
        et_preds = np.zeros((math.ceil(len(y)*0.25),1), dtype=np.int)
        gp_preds = np.zeros((math.ceil(len(y)*0.25),1), dtype=np.int)


   
        ###############################################################################
        #                 . GaussianProcess   +  Radial-Basis Function                #
        ###############################################################################
        if any(x == 'gp' for x in comb[i]):
          gp_scaler = trial.suggest_categorical("gp_Scaler", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(gp_scaler,X, y)
          a_gp = trial.suggest_loguniform("gp_a", 0.001, 10)
          gp_kern= trial.suggest_int("gp_kern", 1, 15)
          gpkernel = a_gp * RBF(gp_kern)
          gp = GaussianProcessClassifier(kernel=gpkernel, random_state=0, n_jobs = -1).fit(train_x, train_y)
          gp_preds = gp.predict_proba(valid_x)[:,1]

        ###############################################################################
        #                              . Extra Trees                                  #
        ###############################################################################
        if any(x == 'et' for x in comb[i]):
          et_scaler = trial.suggest_categorical("et_Scaler", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(et_scaler,X, y)
          et_md = trial.suggest_int("et_max_depth", 1, 100)
          et_ne = trial.suggest_int("et_ne", 1, 500) #1000
          et = ExtraTreesClassifier(max_depth=et_md, n_estimators = et_ne,
                                     random_state=0).fit(train_x, train_y)
          et_preds = et.predict_proba(valid_x)[:,1]
        ###############################################################################
        #                                 . XGBoost                                   #
        ###############################################################################
        if any(x == 'xgb' for x in comb[i]): 
          xgb_scaler = trial.suggest_categorical("xgb_Scaler2", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(xgb_scaler,X, y)
          xgb_param = {
              "silent": 1,
              "objective": "binary:logistic", # change for multiclass
              "eval_metric": "auc",
              # "num_class": 3, # change this up depending on multiclass | dont use with binary
              "booster": trial.suggest_categorical("booster2", ["gbtree", "gblinear", "dart"]),
              "lambda": trial.suggest_loguniform("lambda2", 1e-8, 1.0),
              "alpha": trial.suggest_loguniform("alpha2", 1e-8, 1.0),
          }
          if xgb_param["booster"] == "gbtree" or xgb_param["booster"] == "dart":
              xgb_param["max_depth"] = trial.suggest_int("max_depth2", 1, 100)
              xgb_param["eta"] = trial.suggest_loguniform("eta2", 1e-8, 1.0)
              xgb_param["gamma"] = trial.suggest_loguniform("gamma2", 1e-8, 1.0)
              xgb_param["grow_policy"] = trial.suggest_categorical("grow_policy2", ["depthwise", "lossguide"])
          if xgb_param["booster"] == "dart":
              xgb_param["sample_type"] = trial.suggest_categorical("sample_type2", ["uniform", "weighted"])
              xgb_param["normalize_type"] = trial.suggest_categorical("normalize_type2", ["tree", "forest"])
              xgb_param["rate_drop"] = trial.suggest_loguniform("rate_drop2", 1e-8, 1.0)
              xgb_param["skip_drop"] = trial.suggest_loguniform("skip_drop2", 1e-8, 1.0)
          # Add a callback for pruning.
          xgb_pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc" )
          xgb_ = xgb.train(xgb_param, dtrain_xbg, evals=[(dvalid_xbg, "validation")], verbose_eval=False, callbacks=[xgb_pruning_callback])
          xgb_preds = xgb_.predict(dvalid_xbg)
          self._xgb = xgb_
        ###############################################################################
        #                          . Light Gradient Boosting                          #
        ###############################################################################
        if any(x == 'gbm' for x in comb[i]):
          gbm_scaler = trial.suggest_categorical("gbm_Scaler2", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(gbm_scaler,X, y)
          gbm_param = {
            'objective': 'binary', # change for multiclass
            'metric': 'auc',
              "verbosity": -1,
              "boosting_type": "gbdt",
              "lambda_l1": trial.suggest_loguniform("lambda_l12", 1e-8, 10), 
              "lambda_l2": trial.suggest_loguniform("lambda_l22", 1e-8, 10),
              "num_leaves": trial.suggest_int("num_leaves2", 2, 256), 
              "feature_fraction": trial.suggest_uniform("feature_fraction2", 0.4, 1.0), 
              "bagging_fraction": trial.suggest_uniform("bagging_fraction2", 0.4, 1.0),
              "bagging_freq": trial.suggest_int("bagging_freq2", 1, 7), 
              "min_child_samples": trial.suggest_int("min_child_samples2", 2, 20), 
          }
          # Add a callback for pruning.
          gbm_pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
          gbm = lgb.train(gbm_param, dtrain_gbm, valid_sets=[dvalid_gbm], verbose_eval=False, callbacks=[gbm_pruning_callback])
          gbm_preds = gbm.predict(valid_x)
          self._gbm = gbm
        ###############################################################################
        #                            . Stacking Strategy                              #
        ###############################################################################

        preds = (gbm_preds + xgb_preds +  \
         + et_preds + \
         gp_preds ) / 3 
        preds = preds[:1][0]


        self.predictions = preds
        auc = average_precision_score(valid_y, preds)
        return auc

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_gbm = self._gbm
            self.best_xgb = self._xgb
            self.fpredictions = self.predictions
            

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # for log error

import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 0-2')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
datacopy = w24_df
x =[]
for col in w24_df.columns:
  n = len(datacopy[col].unique())
  if (15 < n):
       continue
  if (15 > n > 2 ): # making a list of columns that are greater than 2 levels
      y = col
      x.append(y)  

w24X = pd.get_dummies(datacopy, columns = x)

X = w24X
y =  w24['EVENTUAL_ICU']


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # for log error

import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 2-4')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
datacopy = w46_df
x =[]
for col in w46_df.columns:
  n = len(datacopy[col].unique())
  if (15 < n):
       continue
  if (15 > n > 2 ): # making a list of columns that are greater than 2 levels
      y = col
      x.append(y)  

w46X = pd.get_dummies(datacopy, columns = x)

X = w46X
y =  w46['EVENTUAL_ICU']


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # for log error

import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 4-6')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
datacopy = w612_df
x =[]
for col in w612_df.columns:
  n = len(datacopy[col].unique())
  if (15 < n):
       continue
  if (15 > n > 2 ): # making a list of columns that are greater than 2 levels
      y = col
      x.append(y)  

w612X = pd.get_dummies(datacopy, columns = x)

X = w612X
y =  w612['EVENTUAL_ICU']


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # for log error

import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 6-12')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
list1_as_set = set(demo_lst + vitalSigns_lst)
intersection = list1_as_set.intersection(w02X.columns.tolist())
demo_vitalSigns = list(intersection)
X = w02X[demo_vitalSigns]
y =  w02['EVENTUAL_ICU']
import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 0-2')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w24X.columns.tolist())
demo_vitalSigns = list(intersection)
X = w24X.loc[:,demo_vitalSigns]
y =  w24['EVENTUAL_ICU']
import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 2-4')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w46X.columns.tolist())
demo_vitalSigns = list(intersection)
X = w46X.loc[:,demo_vitalSigns]
y =  w46['EVENTUAL_ICU']
import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 4-6')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w612X.columns.tolist())
demo_vitalSigns = list(intersection)
X = w612X.loc[:,demo_vitalSigns]
y =  w612['EVENTUAL_ICU']
import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 6-12')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
list1_as_set = set(demo_lst + lab_lst)
intersection = list1_as_set.intersection(w02X.columns.tolist())
demo_lab = list(intersection)

X = w02X.loc[:,demo_lab]
y =  w02['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 0-2')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w24X.columns.tolist())
demo_lab = list(intersection)

X = w24X.loc[:,demo_lab]
y =  w24['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    

train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 2-4')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w46X.columns.tolist())
demo_lab = list(intersection)

X = w46X.loc[:,demo_lab]
y =  w46['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 4-6')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w612X.columns.tolist())
demo_lab = list(intersection)

X = w612X.loc[:,demo_lab]
y =  w612['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 6-12')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
list1_as_set = set(demo_lst + comorb_lst)
intersection = list1_as_set.intersection(w02X.columns.tolist())
demo_como = list(intersection)

X = w02X.loc[:,demo_como]
y =  w02['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 0-2')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w24X.columns.tolist())
demo_como = list(intersection)

X = w24X.loc[:,demo_como]
y =  w24['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 2-4')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w46X.columns.tolist())
demo_como = list(intersection)

X = w46X.loc[:,demo_como]
y =  w46['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 4-6')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
intersection = list1_as_set.intersection(w612X.columns.tolist())
demo_como = list(intersection)

X = w612X.loc[:,demo_como]
y =  w612['EVENTUAL_ICU']


import optuna
objective = Objective()

# Setting SEED 
from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)

study = optuna.create_study(
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize",
    sampler=sampler
)
study.optimize(objective, n_trials=1000, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',X,y) # just taking the validation set
predictions = objective.fpredictions
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)

print('WINDOW 6-12')
print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')
def boxplot(data,col,title):
    sns.set_style('ticks')
    f, ax = plt.subplots(figsize=(32, 18))
#     ax.set_xscale("log")
    sns.set(font_scale=0.8)
    sns.boxplot(data=data[col],palette="vlag", orient="h")
    plt.title(title, size= 14)
    ax.yaxis.grid(True)
    sns.despine()
    ax.set(ylabel='')
    plt.show()
    
boxplot(w02,vitalSigns_lst,'Vital Signs')
list1_as_set = set(lab_lst)
intersection = list1_as_set.intersection(w02X.columns.tolist())
lab = list(intersection)
boxplot(w02,lab,'Lab')
sns.set(style="ticks")
fig, axs = plt.subplots(1,len(demo_lst))
fig.set_size_inches(32, 18)
i=0
for col in demo_lst:
    sns.set_style("white")
    sns.countplot(col, data=df, palette="vlag",  ax=axs[i])
    i+=1
sns.despine()
plt.show()
def make_corr(data,subgroup,title=''):
    sns.set(font_scale=0.8)
    cols = subgroup  #columns gohere
    plt.figure(figsize=(32,18)) # plotting heapmap
    sns.heatmap(data[cols].corr(), cmap='RdBu_r', annot=False, center=0.0)
    if title!='': plt.title(title) # title based on input 
    plt.show()

make_corr(w02, w02.columns.tolist(),'All Column Corr')
