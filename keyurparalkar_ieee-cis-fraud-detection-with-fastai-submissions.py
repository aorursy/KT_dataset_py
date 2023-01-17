# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install fastai==0.7.0
from fastai import *

from fastai.structured import *
import pathlib
from sklearn.model_selection import train_test_split

from sklearn.ensemble import *
PATH = '/kaggle/input/ieee-fraud-detection/'

working_path = '/kaggle/working/'



path = pathlib.Path(PATH)

path_w = pathlib.Path(working_path)
!ls {PATH}
# !head -n 10000 {path}/train_transaction.csv > {path_w}/sample_train_transaction.csv

# !head -n 10000 {path}/train_identity.csv > {path_w}/sample_train_identity.csv
# trans = pd.read_csv(path_w/'sample_train_transaction.csv')

# trans
# iden = pd.read_csv(path_w/'sample_train_identity.csv')

# iden
# len(trans.columns),len(iden.columns)
# data = pd.merge(left=trans,right=iden,on='TransactionID',how='left')
# train_cats(data)
# df_trn, y_trn, nas = proc_df(data,y_fld='isFraud')
# def split_vals(a,n): return a[:n],a[n:]



# n_valid = 1200

# n_train = len(data) - n_valid

# X_train, X_valid = split_vals(df_trn,n_train)

# y_train, y_valid = split_vals(y_trn,n_train)

# raw_train, raw_valid = split_vals(data,n_train)
# raw_train.head()
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_auc_score, roc_curve, auc
# def print_score(m):

#     results = [roc_auc_score(y_train,m.predict(X_train)),roc_auc_score(y_valid,m.predict(X_valid)),

#                m.score(X_train,y_train),m.score(X_valid,y_valid)]

#     print("Scores = ",results)
# set_rf_samples(5000)
# model = RandomForestRegressor(n_estimators=10,min_samples_leaf=3,max_features=0.5,n_jobs=-1)

# model.fit(X_train,y_train)

# print_score(model)
# %time preds = np.stack([t.predict(X_valid) for t in model.estimators_])
# np.mean(preds[:,0]),np.std(preds[:,0])
# fi = rf_feat_importance(model,df_trn)

# def plot_fi(fi): return fi.plot('cols','imp','barh',figsize=(12,10),legend=True)

# plot_fi(fi[:25])
# imps1 = fi[fi['imp'] > 0.015]['cols'].tolist()

# len(imps1)
# imps = ['TransactionID'] + list(filter(lambda x: x!='TransactionID',imps1))
# len(imps)
# df_trn_imps = df_trn.copy()
# #splitting the data again into train and validation set:

# X_train, X_valid = split_vals(df_trn_imps, n_train)

# y_train, y_valid = split_vals(y_trn,n_train)
# set_rf_samples(5000)
# model_imps = RandomForestRegressor(n_estimators=10,min_samples_leaf=3,max_features=0.5,n_jobs=-1)

# model_imps.fit(X_train,y_train)

# print_score(model_imps)
# %time preds_imps = np.stack([t.predict(X_valid) for t in model_imps.estimators_])

# np.mean(preds_imps[:,0]),np.std(preds_imps[:,0])
#Training this on a full scale dataset



#Reading the dataset

trans_full = pd.read_csv(path/'train_transaction.csv')

iden_full = pd.read_csv(path/'train_identity.csv')

data = pd.merge(left=trans_full,right=iden_full,on='TransactionID',how='left')



#converting object data to categorical data:

train_cats(data)
def split_vals(a,n): return a[:n],a[n:]



df_trn, y_trn, nas = proc_df(data,y_fld='isFraud')



n_valid = 118108

n_train = len(df_trn) - n_valid



X_train, X_valid = split_vals(df_trn,n_train)

y_train, y_valid = split_vals(y_trn, n_train)

train_raw, valid_raw = split_vals(data, n_train)
imps = ['TransactionID','C12','C8','card1','C1','TransactionAmt','C13','TransactionDT']

X_train = X_train[imps]

X_valid = X_valid[imps]

X_train.columns,X_valid.columns
X_train.shape
set_rf_samples(50000)
# estimators_list = [80,100,120,200,300,400,500]

# min_samples_leaf = 3

# max_features = 1
# def return_score(est_list,min_sampls,max_fts):

#     total_findings = []

#     for n,est in enumerate(est_list):

#         m = RandomForestRegressor(n_estimators=est,min_samples_leaf=min_sampls,max_features=max_fts,n_jobs=-1,oob_score=True)

#         m.fit(X_train,y_train)

#         results = [roc_auc_score(y_train,m.predict(X_train)),roc_auc_score(y_valid,m.predict(X_valid)),

#                    m.score(X_train,y_train),m.score(X_valid,y_valid)]

#         print(f'Estimator {n}: Scores = {results}, OOB_SCORE = {m.oob_score_}')

#         total_findings.append(results)

        

#     return total_findings
# def plot_validation_graph(e_list,min_spl,max_fts):

#     tf_findings = return_score(e_list,min_spl,max_fts)

#     valid_acc = [t[-1] for t in tf_findings]

#     valid_auc_score = [t[1] for t in tf_findings]

    

#     fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,7))



#     ax1.plot(estimators_list,valid_acc)

#     ax1.set_xlabel('Number of estimators')

#     ax1.set_ylabel('Mean accuracy')

#     ax1.set_title('Validation set accuracy over estimators')



#     ax2.plot(estimators_list,valid_auc_score)

#     ax2.set_xlabel('Number of estimators')

#     ax2.set_ylabel('roc_auc_score')

#     ax2.set_title('Validation set roc_auc_score over estimators')

    
# plot_validation_graph(estimators_list,3,1)
# plot_validation_graph(estimators_list,2,1)
# plot_validation_graph(estimators_list,1,1)
# plot_validation_graph(estimators_list,2,0.5)
# plot_validation_graph(estimators_list,1,0.5)
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_auc_score, roc_curve, auc
def print_score(m):

    results = [roc_auc_score(y_train,m.predict(X_train)),roc_auc_score(y_valid,m.predict(X_valid)),

               m.score(X_train,y_train),m.score(X_valid,y_valid)]

    if(m.oob_score_):

        print(f'Score = {results}, OOB_SCORE = {m.oob_score_}')

    else:

        print("Score = ",results)
#Based on above analysis let us take mni_samples_leaf = 3 and max_features = 1 with n_estimators = 40

m = RandomForestRegressor(n_estimators=800,min_samples_leaf=3,max_features=1,n_jobs=-1,oob_score=True)

m.fit(X_train,y_train)

print_score(m)
# m_best_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

#                        max_depth=None, max_features=1, max_leaf_nodes=None,

#                        min_impurity_decrease=0.0, min_impurity_split=None,

#                        min_samples_leaf=3, min_samples_split=2,

#                        min_weight_fraction_leaf=0.0, n_estimators=800,

#                        n_jobs=None, oob_score=False, random_state=None,

#                        verbose=0, warm_start=False)
# m_best_model.fit(X_train,y_train)

# print_score(m_best_model)
# print_score(m_best_model)
# %time preds_new = np.stack([t.predict(X_valid) for t in m.estimators_])

# np.mean(preds_new[:,0]),np.std(preds_new[:,0])
# X_valid.shape, preds_new.shape
# valid_cpy = valid_raw.copy()
# valid_cpy['preds'] = np.mean(preds_new,axis=0)

# valid_cpy['preds_std'] = np.std(preds_new,axis=0)
# valid_cpy.groupby('C12').mean()
# import pprint

# pp = pprint.PrettyPrinter(indent=4)
# rf = RandomForestRegressor()
# pp.pprint(rf.get_params())
# from sklearn.model_selection import RandomizedSearchCV
# random_search_grid = {

#     'n_estimators':[x for x in range(200,2001,200)],

#     'min_samples_leaf':[0.5,1,3,5,8,10],

#     'max_features':[0.5,1,3,5,8,10]

# }
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_search_grid, n_iter=10,cv=3,verbose=2,random_state=42,n_jobs=-1)
# rf_random.fit(X_train,y_train)
#Reading test set:

t_trans = pd.read_csv(path/'test_transaction.csv')

t_iden= pd.read_csv(path/'test_identity.csv')

test_data = pd.merge(left=t_trans,right=t_iden,on='TransactionID',how='left')



train_cats(test_data)

test_df,t, _ = proc_df(test_data,na_dict=nas)
test_df
test_df_cpy = test_df.copy()
del m_best_model
imps
test_df_cpy = test_df_cpy[imps]

test_df_cpy.columns
test_preds = m.predict(test_df_cpy)
sub_df = pd.DataFrame({'TransactionID':test_df_cpy['TransactionID'],'isFraud': test_preds})
sub_df.to_csv('submission.csv',index=False)
sub_df