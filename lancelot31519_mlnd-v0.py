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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

#import lightgbm as lgb
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

train = pd.read_csv("../input/application_train.csv")
#train = pd.read_csv("data/application_test.csv")
train = train.drop(columns=["SK_ID_CURR"])
df = train.copy()
def create_dummy(df, col_list):
    '''
    parameters : df - input dataframe, col_list - list of columns to be one-hot-encoded
    returns : new df with those fields in col_list one-hot-encoded
    '''
    res = df.copy()
    for col in col_list:
        df[col] = df[col].apply(lambda s:col+"_"+str(s))
        dummy = pd.get_dummies(df[col])
        res = pd.concat([res, dummy], axis=1)
        res = res.drop(columns = [col])
    return res

def encode_data(org_df):
    
    df = org_df.copy()
    df_types = df.dtypes
    list_obj_var = df_types[df_types==np.object].index.tolist()
    list_int_var = df_types[df_types==np.integer].index.tolist()
    list_cat_var = [ele for ele in list_int_var if ele not in ("TARGET","CNT_CHILDREN","DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_ID_PUBLISH","HOUR_APPR_PROCESS_START")]
    list_cat_var = list_cat_var + list_obj_var
    df = create_dummy(df, list_cat_var)
    
    return df

def normalize_data(org_df):
    '''
    parameter : org_df - dataframe
    returns : a new dataframe where all float variables are normalize
    '''
    df = org_df.copy()
    scaler = StandardScaler()
    df_types = df.dtypes
    list_float_var = df_types[df_types==np.float].index
    df[list_float_var] = scaler.fit_transform(df[list_float_var])
    
    return df

def split_res_data(X, y):    
    '''
    parameters: df, dataframe consisting of all training data
    returns : X_train, y_train, X_test, y_test
    ''' 
    
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    
    
    # oversampling with SMOTE
    sm = SMOTE(random_state=12)
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
    X_train = pd.DataFrame(X_train_res)
    X_train.columns = X_test.columns
    y_train = pd.DataFrame(y_train_res)
    y_train.columns = y_test.columns
    
    return X_train, y_train, X_test, y_test
    

df = df.fillna(value=0)
df = encode_data(df)
df = normalize_data(df)

y = df[["TARGET"]]
X = df.drop(columns = ["TARGET"])

X_train, y_train, X_test, y_test = split_res_data(X,y)
best_score = 0
for num_leaves in [15,63,255]:
    for min_data_in_leaves in [10, 100, 1000]:
        for max_depth in [-1, 5, 10]:
            clf = LGBMClassifier(num_leaves=num_leaves, min_data_in_leaves=min_data_in_leaves, max_depth=max_depth)
            clf.fit(X_train,y_train)
            pred_test = clf.predict_proba(X_test)
            score = roc_auc_score(y_test, pred_test[:,1])
            print(num_leaves,min_data_in_leaves,max_depth,score)
            if score > best_score:
                best_score = score
                params = [num_leaves, min_data_in_leaves, max_depth]
ratios = [0.1, 0.5, 0.9]
clf = LGBMClassifier()
train_score = []
test_score = []
for ratio in ratios:
    X, _, y , _ = train_test_split(X_train,y_train,test_size=1.-ratio)
    clf.fit(X,y)
    pred_train = clf.predict_proba(X)
    train_score.append(roc_auc_score(y, pred_train[:,1]))
    pred_test = clf.predict_proba(X_test)
    test_score.append(roc_auc_score(y_test, pred_test[:,1]))  
plt.plot(ratios, train_score, '--', color="#111111",  label="Training score")
#plt.plot(ratios, test_score, color="#111111", label="Cross-validation score")
plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
clf = cv_clf.best_estimator_
clf.fit(X_train, y_train)
pred_test = clf.predict_proba(X_test)
score = roc_auc_score(y_test, pred_test[:,1])
score
clf = LGBMClassifier(num_leaves=15)
clf.fit(X_train, y_train)
pred_test = clf.predict_proba(X_test)
score = roc_auc_score(y_test, pred_test[:,1])
score
param_grid = { 
    'num_leaves': [15,31,63],
    'min_data_in_leaves' : [10,100,1000]
}

clf = LGBMClassifier()
cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
cv_clf.fit(X_train, y_train)
train["TARGET"].plot.hist(title="Distribution of TARGET variable")
train.drop(["TARGET"],axis=1).dtypes.value_counts()
list_float_var = train_types[train_types==np.float].index
train_float = train[list_float_var]
train_float.head()
null_perc = train_float.isnull().sum() / len(train_float)
len(null_perc[null_perc>0.5])
#sns.boxplot(data = train[list_float_var[[5,58]]], orient="h")
#null_perc[null_perc>0.5]
list_int_var = train_types[train_types==np.int].index
train_int = train[list_int_var]
train_int.head()
null_perc = train_int.isnull().sum() / len(train_int)
train_int.head()
train_int.nunique()
#len(null_perc[null_perc>0.1])
#sns.boxplot(data = train[list_float_var[[5,58]]], orient="h")
#null_perc[null_perc>0.5]
list_obj_var = train_types[train_types==np.object].index
train_obj = train[list_obj_var]
train_obj.head()
null_perc = train_obj.isnull().sum() / len(train_int)
#train_obj.head()
#len(null_perc[null_perc>0.1])
#sns.boxplot(data = train[list_float_var[[5,58]]], orient="h")
null_perc[null_perc>0]
df_types = df.dtypes
list_int_var = df_types[df_types==np.integer].index.tolist()
df[list_int_var].head()
train_types = train.dtypes
list_float_var = train_types[train_types==np.integer].index
train_unique_count=train[list_float_var].apply(lambda x: len(x.unique()))
data = train_unique_count.value_counts()
#clf = RandomForestClassifier()
#clf = XGBClassifier()
clf = LGBMClassifier()

clf.fit(X_train, y_train)
pred_test = clf.predict_proba(X_test)
score = roc_auc_score(y_test, pred_test[:,1])
score
df_types = df.dtypes
list_obj_var = df_types[df_types==np.object].index.tolist()
list_int_var = df_types[df_types==np.integer].index.tolist()
list_cat_var = [ele for ele in list_int_var if ele not in ("CNT_CHILDREN","DAYS_BIRTH")]
list_cat_var = list_cat_var + list_obj_var
df[list_cat_var].head()
from hyperopt import fmin, tpe, hp

def score_lgb(params):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    para = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': params["num_leaves"],
        'learning_rate': params["learning_rate"],
        'feature_fraction': 0.9, #0.8
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    gbm = lgb.train(para,lgb_train)
    pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    score = roc_auc_score(y_test, pred_test)
    return score

space_lgb = {
        'num_leaves': hp.choice('num_leaves', range(8,256)),
        'learning_rate' : hp.choice('learning_rate', [0.01, 0.3, 0.1])
        }
        

def objective(params):
    print(params)
    score = score_lgb(params)
    return -score

best = fmin(fn=objective, space=space_lgb, algo=tpe.suggest, max_evals=50)
print("best:", best)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space_rf = {
    'n_estimators': hp.choice('n_estimators', range(25,500)),
    'max_depth': hp.choice('max_depth', range(1,30))
}
    
def objective(params):
    print(params)
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    score = roc_auc_score(y_test, pred_test)
    return -score
    #return {"loss":-score, "status": STATUS_OK}


best = fmin(fn=objective, space=space_rf, algo=tpe.suggest, max_evals=3)
print("best:", best)
from hyperopt import fmin, tpe, hp

def score_xgb(params):
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    pred_test = clf.predict(X_test)
    score = roc_auc_score(y_test, pred_test)
    return score

space_xgb = {
        'max_depth': hp.choice('max_depth', [2,3,4,5]),
        'learning_rate' : hp.choice('learning_rate', [0.01, 0.3, 0.1])
        }
        #'min_child_weight': hp.quniform ('min_child', 1, 20, 1),
        #'subsample': hp.uniform ('subsample', 0.8, 1),
        #'n_estimators' : hp.choice('n_estimators', np.arange(1000, 10000, 100, dtype=int)),
        #'gamma' : hp.quniform('gamma', 0, 1, 5),
        #'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)
        

def objective(params):
    print(params)
    score = score_xgb(params)
    return -score

best = fmin(fn=objective, space=space_xgb, algo=tpe.suggest, max_evals=3)

print("best:", best)
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

clf = XGBClassifier()

grid_param = {  
    "max_depth": [2,3,4,5],
    "learning_rate": [0.01, 0.3, 0.1 ]
}

gd_sr = GridSearchCV(estimator=clf,  
                     param_grid=grid_param,
                     scoring='roc_auc',
                     cv=5,
                     n_jobs=-1)

gd_sr.fit(X_train, y_train)  
best_parameters = gd_sr.best_params_  
print(best_parameters) 
