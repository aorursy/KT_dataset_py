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
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from sklearn.utils import resample


# matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
# enable static images of your plot embedded in the notebook
%matplotlib inline 

# seaborn
import seaborn as sns
sns.set() #apply the default default seaborn theme, scaling, and color palette

import lightgbm as lgb
import xgboost as xgb


import catboost
from catboost import *
from catboost import datasets
from catboost import Pool
(df_train, df_test) = catboost.datasets.amazon()
df_train.head()
df_train.rename(columns={"ACTION": "TARGET"}, inplace= True)
df_test.rename(columns={"ACTION": "TARGET"},  inplace= True)
df_train['TARGET'].value_counts()
numerical_features = list(df_train.select_dtypes([np.number]).columns)
categorical_features = list(set(df_train.columns).difference(set(numerical_features)))
categorical_features

df_train.shape
df_train.describe()
df_train.info()
df_test.shape
df_test.describe()
df_train['TARGET'].value_counts()
df_train.isna().sum()
df_test.isna().sum()
Features = list(df_train.columns)
Features.remove('TARGET')
cat_features = list(range(df_train.drop("TARGET",1).shape[1]))
print(cat_features)
our_x_train = df_train[Features]
our_y_train = df_train['TARGET']
our_x_test = df_test[Features]
X_train, X_test, y_train, y_test = train_test_split(our_x_train,
                                                    our_y_train,
                                                    test_size=0.33,
                                                    random_state=17)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# checking the label balance in dataset
im_check = df_train.TARGET.value_counts()

print("class 1:", im_check[1]/sum(im_check))
print("class 0:", im_check[0]/sum(im_check))

display(im_check)
# Separate classes
class_0 = df_train[df_train.TARGET == 0]
class_1 = df_train[df_train.TARGET == 1]

# Downsample majority class
down_1_class = resample(class_1,
                        n_samples=1897,    # to match minority class
                        random_state=10)  # reproducible results


# Combine minority class with downsampled majority class
down_df = pd.concat([down_1_class, class_0])

# Join together class 0's target vector with the downsampled class 1's target vector
down_df.TARGET.value_counts()


# downsampled_df
X_dtrain, X_dvalid, y_dtrain, y_dvalid = train_test_split(
    down_df.drop('TARGET', axis=1), down_df.TARGET, train_size=0.8, random_state=950)


def plot_confusion_matrix(actual_val, pred_val, title=None):
    """
    pd.crosstab() - crosstab function builds a cross-tabulation table that can show the 
                    requency with which certain groups of data appear.
    
    """
    confusion_matrix = pd.crosstab(actual_val, pred_val,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    
    plot = sns.heatmap(confusion_matrix, annot=True, fmt=',.0f')
    
    if title is None:
        pass
    else:
        plot.set_title(title)
        
    plt.show()
print("LightGBM version:", lgb.__version__)
def run_lgb(X_train, X_test, y_train, y_test, finetuned_params, title):
    FIXED_PARAMS={'objective': 'binary',
              'metric': 'auc',
              'boosting':'gbdt',
              'num_boost_round':300,
              'verbosity' : -1}
    
    params={**FIXED_PARAMS, **finetuned_params}
    lgtrain = lgb.Dataset(X_train, label= y_train)
    lgval = lgb.Dataset(X_test, label=y_test)
    evals_result = {}
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    
    print("Accuracy", accuracy_score(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred,title)
   
    lgb.plot_importance(booster = model, title =title)
    
params_downsampling = {}
run_lgb(X_dtrain, X_dvalid, y_dtrain, y_dvalid,params_downsampling,"Using Downsampled Dataset" )

params_class_imbalance = {'is_unbalance':True}
run_lgb(X_train, X_test, y_train, y_test,params_class_imbalance,"Using is_unbalance parameter set to True" )


params_scale_pos_weight = {'scale_pos_weight':im_check[0]/im_check[1]}
run_lgb(X_train, X_test, y_train, y_test,params_scale_pos_weight,"Using scale_pos_weight Parameter" )
params_scale_pos_weight = {'scale_pos_weight':im_check[0]/im_check[1]}
run_lgb(X_train, X_test, y_train, y_test,params_scale_pos_weight,"Using scale_pos_weight Parameter" )
params_cat_features = {'cat_feature' : cat_features,
                      'scale_pos_weight':im_check[0]/im_check[1]}
run_lgb(X_train, X_test, y_train, y_test,params_class_imbalance,"Using LightGBMs Categorical Features" )

def run_lgb_gridsearch(X_train, X_test, y_train, y_test,  our_x_test):
    SEARCH_PARAMS = {'learning_rate': 0.4,
                 'max_depth': 15,
                 'num_leaves': 20,
                 'feature_fraction': 0.8,
                 "bagging_fraction" : 0.6,
                 "bagging_frequency" : 6,
                 "bagging_seed" : 42,
                 'subsample': 0.2}

    FIXED_PARAMS={'objective': 'binary',
              'metric': 'auc',
              'is_unbalance':True,
              'early_stopping_rounds':30,
              'verbosity' : -1}
    
    params = {'metric':FIXED_PARAMS['metric'],
             'objective':FIXED_PARAMS['objective'],
             'verbosity' :FIXED_PARAMS['verbosity'],
             'is_unbalance':FIXED_PARAMS['is_unbalance'],
             **SEARCH_PARAMS}
    
    grid = {'learning_rate': [0.03, 0.1],
        "num_leaves" : [20,30,40],
        'max_depth':  [4, 6, 10],
        'boosting':['gbdt', 'dart','goss'],
        'num_boost_round':[100,200,300],
        }
    

    model = lgb.LGBMClassifier(**params)
    
    grid = GridSearchCV(model, grid,
                    verbose=0,
                    cv=3,
                    n_jobs=2)
    # Run the grid
    grid.fit(X_train,y_train)
    print(grid)
    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)

    # Using parameters already set above, replace in the best from the grid search
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['max_depth'] = grid.best_params_['max_depth']
    params['boosting'] = grid.best_params_['boosting']
    params['num_boost_round'] = grid.best_params_['num_boost_round']

    model.fit(X_train,y_train)
    pred_test_y = model.predict(our_x_test)
    
    
    #pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model
pred_test, model = run_lgb_gridsearch(X_train, X_test, y_train, y_test, our_x_test)
print("LightGBM Training Completed...")
model
dftrainLGB = lgb.Dataset(data = X_train, label = y_train)

params = {'objective': 'binary'}


lgb_cv_results = lgb.cv(
        params,
        dftrainLGB,
        num_boost_round=100,
        nfold=3,
        metrics='mae',
        early_stopping_rounds=10,
        stratified=False
        )

lgb_cv_results
print("XGBoost version:", xgb.__version__)
def run_xgb(X_train, X_test, y_train, y_test, finetuned_params, title):
    
    FIXED_PARAMS={'objective':'binary:logistic',
              'eval_metric': 'auc',
              'num_boost_round':300,
              'eta': 0.001,
              'max_depth': 10, 
              'subsample': 0.6, 
              'colsample_bytree': 0.6,
              'alpha':0.001,
              'random_state': 42, 
              'silent':True}

    params={**FIXED_PARAMS, **finetuned_params}
    model_xgb = xgb.XGBClassifier(**params)
    #model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    model_xgb.fit(X_train, y_train)
    y_pred = model_xgb.predict(X_test)
    print("Accuracy", accuracy_score(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred,title)
    xgb.plot_importance(model_xgb)
    plt.show()
    
    
    
params_downsampling = {}
run_xgb(X_dtrain, X_dvalid, y_dtrain, y_dvalid,params_downsampling,"Using Downsampled Dataset" )

params_scale_pos_weight = {'scale_pos_weight':im_check[0]/im_check[1]}
run_xgb(X_train, X_test, y_train, y_test,params_scale_pos_weight,"Using scale_pos_weight Parameter" )
def run_xgb_gridsearch(X_train, X_test, y_train, y_test,  our_x_test):
    params={'objective':'binary:logistic',
              'eval_metric': 'auc',
              'num_boost_round':300,
              'eta': 0.001,
              'max_depth': 10, 
              'alpha':0.001,
              'random_state': 42, 
              'silent':True}
    
    grid = {'learning_rate': [0.03, 0.1],
        "num_leaves" : [20,30,40],
        'max_depth':  [4, 6, 10],
        'num_boost_round':[100,200,300],
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
        }
    

    
    model_xgb = xgb.XGBClassifier(**params)
    
    grid = GridSearchCV(model_xgb, grid,
                    verbose=0,
                    cv=3,
                    n_jobs=2)
    # Run the grid
    grid.fit(X_train,y_train)
    print(grid)
    # Print the best parameters found
    print(grid.best_params_)
    print(grid.best_score_)

    # Using parameters already set above, replace in the best from the grid search
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['max_depth'] = grid.best_params_['max_depth']
    params['num_boost_round'] = grid.best_params_['num_boost_round']
 

    model_xgb.fit(X_train,y_train)
    pred_test_y = model_xgb.predict(our_x_test)
    
    return   pred_test_y ,model_xgb
pred_test_y ,model= run_xgb_gridsearch(X_train, X_test, y_train, y_test, our_x_test)
print("XGBoost Training Completed...")
model

data_dmatrix = xgb.DMatrix(X_train, y_train)
params = {'objective':'binary:logistic','colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="auc", as_pandas=True, seed=123)

cv_results
