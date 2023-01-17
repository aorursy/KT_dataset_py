# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=(16,9)

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold

from sklearn.linear_model import LinearRegression,ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor,LGBMClassifier

from scipy.stats import boxcox,norm,skew



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sub=pd.read_csv('../input/sample_submission.csv')
train.head()
train.target.value_counts()
weights={0:1,1:1.8}
train_df=train.drop(columns=['id','target'],axis=1)

test_df=test.drop(columns=['id'],axis=1)

target=train['target']
from sklearn.preprocessing import StandardScaler,MinMaxScaler

sc=StandardScaler()

test_df=sc.fit_transform(test_df)

train_df=sc.fit_transform(train_df)
train_df.shape,test_df.shape
lgbm=LGBMClassifier(class_weight=weights)
kf=StratifiedKFold(n_splits=5,shuffle=True)
params={'max_depth':[5,4,6,7,8],'n_estimators':[200,300,500,1000],'learning_rate':[0.1,0.5,1,2,4,5,10],'num_leaves':[5,3,4,6]}
gr=GridSearchCV(cv=kf,estimator=lgbm,error_score='roc_auc',verbose=3,param_grid=params,n_jobs=-1)
gr.fit(train_df,target)