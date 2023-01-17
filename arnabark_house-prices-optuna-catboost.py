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

# # from multiprocessing import Pool

# nltk.download("stopwords")

# nltk.download("punkt")

import statsmodels.api as sm

from statsmodels.formula.api import ols

import time
Train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

Test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
Train.shape
pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')


Train['MSSubClass']=Train['MSSubClass'].astype('int')

Train['YrSold']=2020-Train['YrSold']

Train['MoSold']=Train['MoSold'].astype('str')

Train['GarageYrBlt']=2020-Train['GarageYrBlt']

Train['YearBuilt']=2020-Train['YearBuilt']

Train['YearRemodAdd']=2020-Train['YearRemodAdd']
Train['YrSold'].unique()


Test['MSSubClass']=Test['MSSubClass'].astype('int')

Test['YrSold']=2020-Test['YrSold']

Test['MoSold']=Test['MoSold'].astype('str')

Test['GarageYrBlt']=2020-Test['GarageYrBlt']

Test['YearBuilt']=2020-Test['YearBuilt']

Test['YearRemodAdd']=2020-Test['YearRemodAdd']
[print(i,Train[i].unique()[:10],len(Train[i].unique()),Train[i].dtype) for i in Train.columns]
#log transform



#log transform skewed numeric features:

numeric_feats = Train.dtypes[Train.dtypes == ("float" or "int") ].index



skewed_feats = Train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

print(skewed_feats)

Train[skewed_feats] = np.log1p(Train[skewed_feats])

Test[skewed_feats] = np.log1p(Test[skewed_feats])
Train=Train.fillna(0)

Test=Test.fillna(0)
X_t=Train[Train.columns[Train.columns!='SalePrice']]

y_t=Train['SalePrice']

X_test=Test
from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X_t,y_t,test_size=0.2,shuffle=True)
categorical_features_indices= np.where(X_test.dtypes == np.object)[0]

Train_set=Pool(X_train, y_train,cat_features=categorical_features_indices)



Eval_set=Pool(X_valid, y_valid,cat_features=categorical_features_indices)

def objective(trial):

    param = {

        'iterations':500,

        'learning_rate':0.05,

        'use_best_model':True,

        'od_type' : "Iter",

        'od_wait' : 100,

#         'random_seed': 240,

#          "scale_pos_weight":trial.suggest_int("scale_pos_weight", 1, 10),

        "depth": trial.suggest_int("max_depth", 2, 10),

        "l2_leaf_reg": trial.suggest_loguniform("lambda", 1e-8, 100),

          'eval_metric':trial.suggest_categorical("loss_function",['MAE','RMSE']),

         'one_hot_max_size':1024

        }



    # Add a callback for pruning.

    model=CatBoostRegressor(**param)

    print(param)

    model.fit(Train_set,eval_set=Eval_set,plot=False,verbose=False)

    pred=model.predict(Pool(X_valid,cat_features= np.where(X_valid.dtypes == np.object)[0]))

    mae=sklearn.metrics.mean_absolute_error(pred,y_valid)

    



    return mae


import sklearn





# study = optuna.create_study()

# study.optimize(objective, n_trials=100)
# study.best_params


param={'one_hot_max_size':1024,'iterations': 2000, 'learning_rate': 0.05, 'use_best_model': True, 'od_type': 'Iter', 'od_wait': 100, 'depth': 7, 'l2_leaf_reg': 0.13679400187948992,'loss_function':'MAE'}

# model1=CatBoostRegressor(loss_function='MAE',use_best_model= True, od_type= 'Iter',od_wait=100)

model1=CatBoostRegressor(**param)

model1.fit(Train_set,eval_set=Eval_set,plot=True,verbose=False)

pred=model1.predict(Pool(X_test,cat_features= np.where(X_test.dtypes == np.object)[0]))
X_test['SalePrice']=pred
X_test[['Id','SalePrice']].to_csv('submission.csv',index=False)