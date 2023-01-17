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
w02.loc[:,['GENDER', 'AGE_ABOVE65', 'ICU','ICU_W24','ICU_W46','ICU_W612','ICU_Wa12']].head(10)
# FIRST REMOVE ICU 1 FROM WINDOW 0-2
w02 = w02[w02.ICU == 0]

# NEW TARGET "NOT ICU"
w02['temp'] = w02.loc[:,['ICU','ICU_W24','ICU_W46','ICU_W612','ICU_Wa12']].sum(axis=1) 
def label_not_icu(x):
    if (x['temp'] == 0):
        val = 1 
    elif (x['temp'] > 0):
        val = 0
    return val

w02['NOT_ICU'] = w02.apply(label_not_icu, axis=1)

# REMOVE UNWANTED COLUMNS 
w02 = w02.drop(['temp', 'ICU','ICU_W24','ICU_W46','ICU_W612','ICU_Wa12'], axis = 1)
w02.shape
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
combos = list(combinations(w02.columns,2))

# Running to see if any of them are identical
identical_cols = []

for col in np.arange(0,len(combos),1):
    x = [combos[col][0],combos[col][1]]
    if is_one_to_one(w02,x) == True:
         identical_cols.append(combos[col][0])
            
all_cols = [x for x in w02.columns if x not in identical_cols]
w02 = w02.loc[:, all_cols]
w02 = w02.drop('WINDOW', axis = 1)
w02.info(verbose=True)
w02['NOT_ICU'].value_counts()
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
datacopy = w02
x =[]
for col in w02.columns:
    n = len(datacopy[col].unique())
    if (15 < n):
        continue
    if (15 > n > 2 ): # making a list of columns that are greater than 2 levels
        y = col
        x.append(y)      
#     elif (n == 2): # a categorical descriptive feature has only 2 levels
#         datacopy[col] = pd.get_dummies(datacopy[col], drop_first=True)

datacopy = pd.get_dummies(datacopy, columns = x)
datacopy.shape
X = datacopy.drop('NOT_ICU', axis = 1)
DX =  datacopy.drop('NOT_ICU', axis = 1)
y =  datacopy['NOT_ICU']
dy =  datacopy['NOT_ICU']
def brute_force_feat(X,DX):
  X = X.loc[:,~X.columns.duplicated()]

  from sklearn.impute import SimpleImputer
  imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp_mean.fit(X)
  X = imp_mean.transform(X)
  DX = imp_mean.transform(DX)

  print('Dimensionality Features')
  # DIMENSIONALITY FEATURES
  # 2 is usually a good size, however higher may result in less reward for time taken
  X_TSNE = TSNE(n_components=2).fit_transform(X)
  X_DBSCAN = DBSCAN(eps=3, min_samples=2).fit(X)
  X_PCA = PCA(n_components=2).fit_transform(X)
  # Depending on how large the dataset is increase or decrease n 
  X_KNN64, indices = NearestNeighbors(n_neighbors=64, algorithm='ball_tree').fit(X).kneighbors(X)
  X = np.c_[X, X_KNN64] # NUMPY
  X = np.c_[X, X_PCA]
  X = np.c_[X, X_DBSCAN.labels_]
  X = np.c_[X, X_TSNE]
  X = np.where(np.isnan(X), 0, X)


  # Selecting best 50 columns 
  from sklearn.ensemble import RandomForestClassifier

  # NUMPY
  print('No. Features:', np.size(X,1))
  np.random.seed(42)
  X = np.nan_to_num(X.astype(np.float32)) # prevents too large values error

  # SX IS A SAMPLE OF X
  SX = X
  sy = dy # other methods if this gives error #.values.ravel() #np.array(dy)[idx.astype(int)] #dy[idx]
  
  model_rfi = RandomForestClassifier(n_jobs = -1, class_weight = 'balanced_subsample', random_state=42)

  model_rfi.fit(SX, sy)
  print('Done!')

  return X, DX, model_rfi

# CX is the new features data | DX is the original data  | Model_rfi contains the feature importance
CX, DX, model_rfi = brute_force_feat(X,DX)
def scaler_fuc(scaler, f, CX, DX):

  # order by most important features and take the f most important features
  fs_indices_rfi = np.argsort(model_rfi.feature_importances_)[::-1][0:f]
  CX = CX[:,fs_indices_rfi]
  X = np.c_[DX, CX]
  # delete duplicates
  _, unq_col_indices = np.unique(X,return_index=True,axis=1)
  X = X[:,unq_col_indices]

  train_x, valid_x, train_y, valid_y = train_test_split(X, dy,
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
models = ['gp', 'et', 'xgb', 'gbm'] # when you want to try all of them/ iteration: 1 
comb = list(combinations(models, 3))

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
        gbm_preds = np.zeros((math.ceil(len(dy)*0.25),1), dtype=np.int)
        xgb_preds = np.zeros((math.ceil(len(dy)*0.25),1), dtype=np.int)
        et_preds = np.zeros((math.ceil(len(dy)*0.25),1), dtype=np.int)
        gp_preds = np.zeros((math.ceil(len(dy)*0.25),1), dtype=np.int)


   
        ###############################################################################
        #                 . GaussianProcess   +  Radial-Basis Function                #
        ###############################################################################
        if any(x == 'gp' for x in comb[i]):
          gp_f = trial.suggest_int("gp_features", 0, 20)
          gp_scaler = trial.suggest_categorical("gp_Scaler", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(gp_scaler, gp_f, CX, DX)
          a_gp = trial.suggest_loguniform("gp_a", 0.001, 10)
          gp_kern= trial.suggest_int("gp_kern", 1, 15)
          gpkernel = a_gp * RBF(gp_kern)
          gp = GaussianProcessClassifier(kernel=gpkernel, random_state=0, n_jobs = -1).fit(train_x, train_y)
          gp_preds = gp.predict_proba(valid_x)[:,1]

        ###############################################################################
        #                              . Extra Trees                                  #
        ###############################################################################
        if any(x == 'et' for x in comb[i]):
          et_f = trial.suggest_int("et_features", 0, 20)
          et_scaler = trial.suggest_categorical("et_Scaler", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(et_scaler,et_f, CX, DX)
          et_md = trial.suggest_int("et_max_depth", 1, 100)
          et_ne = trial.suggest_int("et_ne", 1, 500) #1000
          et = ExtraTreesClassifier(max_depth=et_md, n_estimators = et_ne,
                                     random_state=0).fit(train_x, train_y)
          et_preds = et.predict_proba(valid_x)[:,1]
        ###############################################################################
        #                                 . XGBoost                                   #
        ###############################################################################
        if any(x == 'xgb' for x in comb[i]): 
          xgb_f = trial.suggest_int("xgb_features2", 0, 20)
          xgb_scaler = trial.suggest_categorical("xgb_Scaler2", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(xgb_scaler,xgb_f, CX, DX)
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
          gbm_f = trial.suggest_int("gbm_features2", 0, 20)
          gbm_scaler = trial.suggest_categorical("gbm_Scaler2", ['minmax','stand','log'])
          train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc(gbm_scaler,gbm_f, CX, DX)
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
        auc = roc_auc_score(valid_y, preds)
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
study.optimize(objective, n_trials=300, callbacks=[objective.callback]) # change this to 500 + 

print("Best trial:")
trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

best_gbm = objective.best_gbm
best_xgb = objective.best_xgb
from sklearn.metrics  import accuracy_score, auc, roc_curve, precision_recall_curve, roc_auc_score, precision_score, recall_score, average_precision_score
train_x, valid_x, train_y, valid_y, dtrain_gbm, dvalid_gbm, dtrain_xbg, dvalid_xbg = scaler_fuc('log',3, CX, DX) # just taking the validation set
comb[2]
predictions = objective.fpredictions
predictions.shape
len(train_x)
len(valid_x)
accuracy  = accuracy_score(valid_y, predictions >= 0.5)
roc_auc   = roc_auc_score(valid_y, predictions)
precision = precision_score(valid_y, predictions >= 0.5)
recall    = recall_score(valid_y, predictions >= 0.5)
pr_auc    = average_precision_score(valid_y, predictions)


print(f'Accuracy: {round(accuracy,4)}')
print(f'ROC AUC: {round(roc_auc,4)}')
print(f'Precision: {round(precision,4)}')
print(f'Recall: {round(recall,4)}')
print(f'PR Score: {round(pr_auc,4)}')