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
import xgboost as xgb

import matplotlib

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold
dftrain = pd.read_csv('../input/train.csv')

dftrain.head()
dftest = pd.read_csv('../input/test.csv')

dftest.head()
train_cols = dftrain.columns

train_cols = train_cols.drop(['ID_code','target'])

train_cols
Xtrain = dftrain[train_cols]

ytrain = dftrain[['target']]

Xtest = dftest[train_cols]
param = {'max_depth':14, 'eta':1, 'objective':'binary:logistic','eval_metric':'auc','gamma':0.1,'subsample':0.95,'lambda':3,'alpha':5, 'min_child_weight':100,

         'max_delta_step':0,'tree_method':'hist','max_bin':1024,'max_leaves':1000,'grow_policy':'lossguide','feature_selector':'greedy','top_k':0,

         'scale_pos_weight':3.5}



dtrain = xgb.DMatrix(Xtrain, label = ytrain)





param = {'max_depth':16, 'eta':1, 'objective':'binary:logistic','eval_metric':'auc','gamma':0.1,'subsample':0.95,'lambda':3,'alpha':5, 'min_child_weight':100,

         'max_delta_step':0,'tree_method':'hist','max_bin':1024,'max_leaves':1000,'grow_policy':'lossguide','feature_selector':'greedy','top_k':0}

boost_rounds = 50

i=0

kf = StratifiedKFold(n_splits=5, random_state=1)

xgbmodels = []



j = kf.get_n_splits(Xtrain)

XDtrain = xgb.DMatrix(Xtrain)

for train_index, test_index in kf.split(Xtrain,ytrain['target']):

    i=i+1

    print("TRAIN:", train_index, "TEST:", test_index)

    ktrain = Xtrain.loc[train_index]

    kytrain = ytrain['target'].loc[train_index]

    ktest = Xtrain.loc[test_index]

    kytest = ytrain['target'].loc[test_index]

    

    dtrain = xgb.DMatrix(ktrain, label = kytrain)

    dtest = xgb.DMatrix(ktest)

    

    xgbmodel = xgb.train(param, dtrain, num_boost_round=boost_rounds)

    preds = xgbmodel.predict(dtest)

    

    xgbmodels.append(xgbmodel)

    col_name = 'Prediction_'+ str(i)

    ytrain[col_name] = xgbmodel.predict(XDtrain)

    auc = roc_auc_score(kytest,preds)

    print('AUC',auc)   

    
pred_cols = ytrain.columns

pred_cols = pred_cols.drop(['target'])

dtrain = xgb.DMatrix(ytrain[pred_cols], label=ytrain['target'])

xgbmodel = xgb.train(param, dtrain, num_boost_round=boost_rounds)

auc = roc_auc_score(ytrain['target'],xgbmodel.predict(dtrain))

print('AUC',auc)   
i=0

ytest = pd.DataFrame(index=Xtest.index, columns=pred_cols)

dtest = xgb.DMatrix(Xtest)

for i in range(kf.get_n_splits(Xtrain)):

    ytest[pred_cols[i]]=xgbmodels[i].predict(dtest)



dtest = xgb.DMatrix(ytest)

preds = xgbmodel.predict(dtest)
dfprediction = dftest[['ID_code']]
dfprediction['target']=preds
dfprediction.head()
dfprediction.to_csv("submission.csv", index=False)