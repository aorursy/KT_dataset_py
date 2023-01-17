# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import gc
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore',category = DeprecationWarning)
warnings.simplefilter(action='ignore',category = DeprecationWarning)


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, RepeatedKFold, GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN
import category_encoders as ce
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from time import time

import scipy.stats as st
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
from shutil import rmtree

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 


import os
print(os.listdir("../input"))

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings(record=True) as w:
    # Cause all warnings to always be triggered.
    warnings.simplefilter("always")
    # Trigger a warning.
    fxn()
    # Verify some things
    assert len(w) == 1
    assert issubclass(w[-1].category, DeprecationWarning)
    assert "deprecated" in str(w[-1].message)

# Any results you write to the current directory are saved as output.
def dprint(*args, **kwargs):
    print("[{}] ".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")) + \
        " ".join(map(str,args)), **kwargs)

id_name = 'Id'
target_name = 'Target'
# Load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['is_test'] = 0
test['is_test'] = 1
df_all = pd.concat([train, test], axis=0)

dprint('Clean features...')
cols = ['dependency']
for c in tqdm(cols):
    x = df_all[c].values
    strs = []
    for i, v in enumerate(x):
        try:
            val = float(v)
        except:
            strs.append(v)
            val = np.nan
        x[i] = val
    strs = np.unique(strs)

    for s in strs:
        df_all[c + '_' + s] = df_all[c].apply(lambda x: 1 if x == s else 0)

    df_all[c] = x
    df_all[c] = df_all[c].astype(float)
dprint("Done.")
dprint("Extracting features...")
def extract_features(df):
    df['bedrooms_to_rooms'] = df['bedrooms']/df['rooms']
    df['rent_to_rooms'] = df['v2a1']/df['rooms']
    df['rent_to_bedrooms'] = df['v2a1']/df['bedrooms']
    df['tamhog_to_rooms'] = df['tamhog']/df['rooms'] # tamhog - size of the household
    df['tamhog_to_bedrooms'] = df['tamhog']/df['bedrooms']
    df['r4t3_to_tamhog'] = df['r4t3']/df['tamhog'] # r4t3 - Total persons in the household
    df['r4t3_to_rooms'] = df['r4t3']/df['rooms'] # r4t3 - Total persons in the household
    df['r4t3_to_bedrooms'] = df['r4t3']/df['bedrooms']
    df['rent_to_r4t3'] = df['v2a1']/df['r4t3']
    df['v2a1_to_r4t3'] = df['v2a1']/(df['r4t3'] - df['r4t1'])
    df['hhsize_to_rooms'] = df['hhsize']/df['rooms']
    df['hhsize_to_bedrooms'] = df['hhsize']/df['bedrooms']
    df['rent_to_hhsize'] = df['v2a1']/df['hhsize']
    df['qmobilephone_to_r4t3'] = df['qmobilephone']/df['r4t3']
    df['qmobilephone_to_v18q1'] = df['qmobilephone']/df['v18q1']
    

extract_features(train)
extract_features(test)
dprint("Done.")         
from sklearn.preprocessing import LabelEncoder

def encode_data(df):
   
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])

dprint("Encoding Data....")
encode_data(train)
encode_data(test)
dprint("Done...")
def do_features(df):
    feats_div = [('children_fraction', 'r4t1', 'r4t3'), 
                 ('working_man_fraction', 'r4h2', 'r4t3'),
                 ('all_man_fraction', 'r4h3', 'r4t3'),
                 ('human_density', 'tamviv', 'rooms'),
                 ('human_bed_density', 'tamviv', 'bedrooms'),
                 ('rent_per_person', 'v2a1', 'r4t3'),
                 ('rent_per_room', 'v2a1', 'rooms'),
                 ('mobile_density', 'qmobilephone', 'r4t3'),
                 ('tablet_density', 'v18q1', 'r4t3'),
                 ('mobile_adult_density', 'qmobilephone', 'r4t2'),
                 ('tablet_adult_density', 'v18q1', 'r4t2'),
                 #('', '', ''),
                ]
    
    feats_sub = [('people_not_living', 'tamhog', 'tamviv'),
                 ('people_weird_stat', 'tamhog', 'r4t3')]

    for f_new, f1, f2 in feats_div:
        df['fe_' + f_new] = (df[f1] / df[f2]).astype(np.float32)       
    for f_new, f1, f2 in feats_sub:
        df['fe_' + f_new] = (df[f1] - df[f2]).astype(np.float32)
    
    # aggregation rules over household
    aggs_num = {'age': ['min', 'max', 'mean'],
                'escolari': ['min', 'max', 'mean']
               }
    aggs_cat = {'dis': ['mean']}
    for s_ in ['estadocivil', 'parentesco', 'instlevel']:
        for f_ in [f_ for f_ in df.columns if f_.startswith(s_)]:
            aggs_cat[f_] = ['mean', 'count']
    # aggregation over household
    for name_, df_ in [('18', df.query('age >= 18'))]:
        df_agg = df_.groupby('idhogar').agg({**aggs_num, **aggs_cat}).astype(np.float32)
        df_agg.columns = pd.Index(['agg' + name_ + '_' + e[0] + "_" + e[1].upper() for e in df_agg.columns.tolist()])
        df = df.join(df_agg, how='left', on='idhogar')
        del df_agg
    # do something advanced above...
    
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id', 'idhogar'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df
    
dprint("Do_feature Engineering....")
train = do_features(train)
test = do_features(test)
dprint("Done....")
dprint("Fill Na value....")
train = train.fillna(0)
test = test.fillna(0)
dprint("Done....")
train.shape,test.shape
cols_to_drop = [
    id_name, 
    target_name,
]
X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train[target_name].values

X.shape,y.shape
# A parameter grid for XGBoost
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='multi:softmax',booster='gbtree',
                    silent=True, nthread=1)

folds = 3
param_comb = 5
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=skf.split(X,y), verbose=0, random_state=1001 )
random_search.fit(X, y)
print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_ * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
y_test = random_search.predict(test)
sub = pd.read_csv("../input/sample_submission.csv")
sub['Target'] = y_test
sub.to_csv("xgb.csv", index= False)
gc.collect()
params = {
        'max_depth': (4, 6),
        'gamma': (0.0001, 0.005),
        'min_child_weight': (1, 2),
        'max_delta_step': (0, 1),
        'subsample': (0.2, 0.4),
        'colsample_bytree': (0.2, 0.4)
        }
xgb = XGBClassifier(learning_rate=0.02, n_estimators=100, objective='multi:softmax',
                    silent=True, nthread=4)
folds = 3
param_comb = 5
skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='accuracy', n_jobs=4, cv=skf.split(X,y), verbose=0,refit=True )
grid.fit(X, y)
print('\n All results:')
print(grid.cv_results_)
print('\n Best estimator:')
print(grid.best_estimator_)
print('\n Best score:')
print(grid.best_score_ * 2 - 1)
print('\n Best parameters:')
print(grid.best_params_)
results = pd.DataFrame(grid.cv_results_)
results.to_csv('xgb-grid-search-results-01.csv', index=False)
gc.collect()

y_test = grid.predict(test)
sub = pd.read_csv("../input/sample_submission.csv")
sub['Target'] = y_test
sub.to_csv("xgb1.csv", index= False)
from skopt import BayesSearchCV
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = XGBClassifier(
        n_jobs = 1,
        objective = 'multi:softmax',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (0, 10),
        'max_depth': (0, 50),
        'max_delta_step': (0, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 100),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    scoring = 'accuracy',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 3,
    n_iter = 10,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
# Fit the model
result = bayes_cv_tuner.fit(X, y, callback=status_print)
gc.collect()
print('\n All results:')
print(bayes_cv_tuner.cv_results_)
print('\n Best estimator:')
print(bayes_cv_tuner.best_estimator_)
print('\n Best score:')
print(bayes_cv_tuner.best_score_ * 2 - 1)
print('\n Best parameters:')
print(bayes_cv_tuner.best_params_)
y_test = bayes_cv_tuner.predict(test)
sub = pd.read_csv("../input/sample_submission.csv")
sub['Target'] = y_test
sub.to_csv("xgb2.csv", index= False)