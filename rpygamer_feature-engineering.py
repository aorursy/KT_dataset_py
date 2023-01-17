# Packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

####################

from sklearn import *

import xgboost as xgb

# Train data 

train = pd.read_csv("../input/train.csv")

train.head()
# test data 

test = pd.read_csv("../input/test.csv")

test.head()
Features=dict()

Features['All']=list(train.columns)

Features['target']=['target']

Features['train']= list(set(train.columns)- set(['id','target']))
X_train=train[Features['train']]

X_test=test[Features['train']]
X_train.shape
X_test.shape
Y=train.target
import random

random.seed( 3 )



def gini(y, pred):

    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)

    g = 2 * metrics.auc(fpr, tpr) - 1

    return g



def gini_xgb(pred, y):

    y = y.get_label()

    return 'gini', gini(y, pred) / gini(y, y)



x1, x2, y1, y2 = model_selection.train_test_split(X_train, train['target'], test_size=0.3, random_state=6)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

params = {'eta': 0.09, 'max_depth': 6, 'objective': 'binary:logistic', 'seed': 16, 'silent': True, 'colsample_bytree': 0.6}

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
train_float = X_train.select_dtypes(include=['float64'])

train_int = X_train.select_dtypes(include=['int64'])
train_float.head()
colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
Features['Reg']=['ps_reg_03','ps_reg_02','ps_reg_01'] # Reg maybe regression !! Why not ? 
X_train.ps_reg_03.describe()
X_train.ps_reg_02.describe()
X_train.ps_reg_01.describe()
Features['FE']=['sum_Reg']
X_train['Sum_Reg']=X_train['ps_reg_03']+X_train['ps_reg_02']+X_train['ps_reg_01']
import random

random.seed( 3 )

x1, x2, y1, y2 = model_selection.train_test_split(X_train, train['target'], test_size=0.3, random_state=6)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

params = {'eta': 0.09, 'max_depth': 6, 'objective': 'binary:logistic', 'seed': 16, 'silent': True, 'colsample_bytree': 0.6}

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
xgb.plot_importance(booster=model,max_num_features=15)
for a in [0,1,5,10,20,100]:

    FE_1='logps_car13_'+str(a)

    print(FE_1)

    X_train[FE_1]=np.log(a+X_train.ps_car_13)
import random

random.seed( 3 )

x1, x2, y1, y2 = model_selection.train_test_split(X_train, train['target'], test_size=0.3, random_state=6)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

params = {'eta': 0.09, 'max_depth': 6, 'objective': 'binary:logistic', 'seed': 16, 'silent': True, 'colsample_bytree': 0.6}

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
xgb.plot_importance(booster=model,max_num_features=15)
X_train.ps_reg_03.value_counts()
X_train.ps_car_13.value_counts()
X_train['Sum_Top_2']=X_train.ps_car_13+X_train.ps_reg_03

X_train['Prod_Top_2']=X_train.ps_car_13*X_train.ps_reg_03

X_train['M_Top_2']=X_train.ps_car_13-X_train.ps_reg_03

X_train['ratio_Top_2']=X_train.ps_car_13 / X_train.ps_reg_03
import random

random.seed( 3 )

x1, x2, y1, y2 = model_selection.train_test_split(X_train, train['target'], test_size=0.3, random_state=6)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

params = {'eta': 0.09, 'max_depth': 6, 'objective': 'binary:logistic', 'seed': 16, 'silent': True, 'colsample_bytree': 0.6}

model = xgb.train(params, xgb.DMatrix(x1, y1), 1000,  watchlist, feval=gini_xgb, maximize=True, verbose_eval=50, early_stopping_rounds=100)
xgb.plot_importance(booster=model,max_num_features=15)