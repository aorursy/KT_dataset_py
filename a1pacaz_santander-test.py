# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import seaborn as sns

import lightgbm as lgb

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.patches as patch

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import zero_one_loss

from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA 

from collections import Counter

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler,SMOTE

# Any results you write to the current directory are saved as output.
# this

tr = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
features = [col for col in tr.columns if col not in ['target', 'ID_code']]

tr_X = tr[features]

tr_y = tr.target

ts_X = test[features]

t_X = pd.concat([tr_X,ts_X],axis = 0,sort = True)
#normalizing data

#tfts = [col for col in test.columns if col not in ['target', 'ID_code']]

#test_X = test[tfts]

#for col in tr_X.columns:

#    tr_X[col] = ((tr_X[col] - tr_X[col].mean()) / tr_X[col].std()).astype('float32')

#for col in test_X.columns:

#   test_X[col] = ((test_X[col] - test_X[col].mean()) / test_X[col].std()).astype('float32')
## Check(and plot) data

## tr['target'].value_counts()

## ros = RandomOverSampler(random_state = 1)

## tr1_X,tr1_y=ros.fit_resample(tr_X,tr_y)

## Counter(tr1_y)

#plt.hist(tr['var_68'])

#sns.kdeplot(tr.loc[tr['target'] == 0]['var_68'],label=0)

#sns.kdeplot(tr.loc[tr['target'] == 1]['var_68'],label=1)

#plt.hist(tr.loc[tr['target'] == 0]['var_68'],label=0)

#plt.hist(tr['var_73'],bins = 1000)

#sns.kdeplot(tr_X['var_33'])

#tr_X['var_33'].iloc[[2,3]]
#randomly change 2 variables

def swapping(x,x1,x2):

    if x == x1:

        return x2

    elif x == x2:

        return x1

    else:

        return x

def appl(x,y,z):

    if x == 1:

        return y

    else:

        return z

for col in t_X.columns:

    count = t_X[col].value_counts()

    t_X[col+'cts'] = t_X[col].map(count)

    t_X[col+'tqs'] = t_X[col+'cts'].apply(lambda x:appl(x,0,1))

    #rdn1 = np.random.normal(loc = tr_X[col].mean(),scale = tr_X[col].std())

    #rdn2 = np.random.normal(loc = tr_X[col].mean(),scale = tr_X[col].std())

    #r1=np.argmin(abs((tr_X[col]-rdn1)))

    #r2=np.argmin(abs((tr_X[col]-rdn2)))

    #a1 = tr_X[col].iloc[r1]

    #a2 = tr_X[col].iloc[r2]

    t_X[col+'tqs'] = t_X[col] * t_X[col+'tqs'] + t_X[col].mean() * (1-t_X[col+'tqs'])

    #tr_X[col+'tqs'] = tr_X[col+'tqs'].apply(lambda x: np.nan if x == 0 else x)

    #tr_X[col+'uqs'] = tr_X[col].map

    #tr_X[col+'rds'] = tr_X[col].apply(lambda x:swapping(x,a1,a2))

    #a = a+sum(tr_X.iloc[col])-sum(tr_X.iloc[col+'rds'])
featurez = [col for col in tr_X.columns if col not in [features]]

tr_X = t_X[:200000]

te_X = t_X[200000:]
#normalizing data

#tfts = [col for col in test.columns if col not in ['target', 'ID_code']]

#test_X = test[tfts]

#for col in tr_X[features]:

#    tr_X[col] = ((tr_X[col] - tr_X[col].mean()) / tr_X[col].std()).astype('float32')

#for col in test_X.columns:

#   test_X[col] = ((test_X[col] - test_X[col].mean()) / test_X[col].std()).astype('float32')

tr_X
## Principle Component Analysis - doesn't work well!

#pca = PCA(n_components=200)

#pca_X = pca.fit_transform(tr_X)

#t_X
#features = [col for col in tr.columns if col not in ['target', 'ID_code']]

#tr_X = tr[features]

#tr_y = tr.target

train_X, val_X, train_y, val_y = train_test_split(tr_X, tr_y, test_size = 0.25, random_state = 0)
#%%time

#model = DecisionTreeClassifier(random_state=0,splitter ='best',max_depth = 10)

#model.fit(train_X,train_y)

#preds = model.predict_proba(val_X)

#preds_y = preds[:,1]

#print('AUC: ', roc_auc_score(val_y, preds_y) )
%%time

#model = RandomForestClassifier(random_state=0,max_depth = 10)

#model.fit(train_X,train_y)

#preds = model.predict_proba(val_X)

#preds_y = preds[:,1]

#print('AUC: ', roc_auc_score(val_y, preds_y) )
%%time

#Logistic Regression

#model = LogisticRegression(class_weight='balanced', penalty='l2',n_jobs = -1, C=0.1,max_iter = 10,verbose=100)

#model.fit(train_X,train_y)

#preds_y = model.predict_proba(val_X)

#preds_y = preds_y[:,1]

#print('AUC: ', roc_auc_score(val_y, preds_y) )
%%time

#Linear Regression

#model = LinearRegression().fit(train_X,train_y)

#preds_y = model.predict(val_X)

#print('AUC: ', roc_auc_score(val_y, preds_y) )

#print('R2: ', model.score(val_X, val_y) )
%%time 

#This method is extremely slow - still needs improvement

# Light GBM

#lgb_train = lgb.Dataset(train_X,train_y)

#lgb_test = lgb.Dataset(val_X,val_y,reference = lgb_train)

#param = {

#    'num_leaves': 20,

#    'verbose':1,

#    'metric':{'auc'},

#    'objective':'binary'

#}

#model = lgb.train(param,lgb_train,num_boost_round = 1000)

#preds_y = model.predict(val_X)

#print('AUC: ', roc_auc_score(val_y,preds_y))
%%time 

#This method is extremely slow - still needs improvement

# Light GBM

lgb_train = lgb.Dataset(train_X,train_y)

lgb_test = lgb.Dataset(val_X,val_y,reference = lgb_train)

param = {

    'num_leaves': 20,

    'verbose':1,

    'metric':{'auc'},

    'objective':'binary'

}

modelx = lgb.train(param,lgb_train,valid_sets = lgb_test,num_boost_round = 2000,early_stopping_rounds = 200,verbose_eval = 200)

preds_y = modelx.predict(val_X)

print('AUC: ', roc_auc_score(val_y,preds_y))
%%time

#Logistic Regression

#modelx = LogisticRegression(class_weight='balanced', penalty='l2', C=0.1,max_iter = 1000,verbose=100)

#modelx.fit(tr1_X,tr1_y)

#preds_y = modelx.predict_proba(val_X)

#print(preds_y)

#preds_y = preds_y[:,1]

#print('AUC: ', roc_auc_score(val_y, preds_y) )
pred = modelx.predict(te_X)

print(pred)

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred

sub.to_csv('submission.csv',index=False)
#tr.describe()