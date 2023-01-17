import numpy as np

import pandas as pd

import qGEL

import xgboost

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit,RandomizedSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, pairwise_distances

from scipy.stats import randint as sp_randint

from scipy.stats import uniform

import random

import scipy.stats as st

import pickle
%%markdown

# Load in data
tr_id=pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

tr_tr=pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', 

                  header=0, 

                  skiprows=lambda i: i>0 and random.random() > 0.25

                 )



te_id=pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

te_tr=pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')



tr_=tr_tr.merge(tr_id, on='TransactionID', how='left')

te_=te_tr.merge(te_id, on='TransactionID', how='left')

tr_.shape, te_.shape
%%markdown 

# Organize data types
time_vars=['TransactionDT', 'D1','D2', 'D3', 'D4', 'D5', 'D6', 

           'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15']

id_cat_vars=list(tr_id.columns[12:39])+['DeviceType', 'DeviceInfo']

tr_cat_vars=['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 

             'card6','addr1', 'addr2','R_emaildomain','P_emaildomain',

             'M1','M2','M3','M4','M5','M6','M7','M8','M9']

removed_vars=['isFraud','TransactionID'] + time_vars + list(tr_.isnull().sum()[tr_.isnull().sum() > .9*len(tr_)].index)

nums=set(tr_.columns).difference(time_vars+id_cat_vars+tr_cat_vars+removed_vars)

len(time_vars), len(id_cat_vars), len(tr_cat_vars), len(nums), len(removed_vars)
%%markdown

# Time variables
time_corr_matrix = tr_[time_vars].corr()

time_upper = time_corr_matrix.where(np.triu(np.ones(time_corr_matrix.shape), k=1).astype(np.bool))

time_to_drop = [column for column in time_upper.columns if any(time_upper[column] > 0.9)]

reduce_time_vars = set(time_vars).difference(time_to_drop)

print(" dropped {0} time varaibles ".format(len(time_vars)-len(reduce_time_vars)))
%%markdown

# Numerical variables
num_corr_matrix = tr_[nums].sample(int(len(tr_)*.10)).corr()

num_upper = num_corr_matrix.where(np.triu(np.ones(num_corr_matrix.shape), k=1).astype(np.bool))

num_to_drop = [column for column in num_upper.columns if any(num_upper[column] > 0.9)]

reduce_num_vars = set(nums).difference(num_to_drop)

print(" dropped {0} numeric varaibles ".format(len(nums)-len(reduce_num_vars)))
%%markdown

# Dummies
to_dummy=tr_[id_cat_vars+tr_cat_vars].nunique()[tr_[id_cat_vars+tr_cat_vars].nunique()<10].index

len(to_dummy)
%%markdown

# Identify and create look-up for high dim variables
lo=tr_.columns.difference(set(time_vars).union(nums).union(id_cat_vars).union(to_dummy).union(removed_vars))
def make_embed(col_name):

    my_samp=tr_[col_name].astype('str').to_frame().sample(10000)

    my_dummies=pd.get_dummies(my_samp[col_name])

    my_emb_, v_t, mb = qGEL.qgel(my_dummies, k=10)

    my_embed=pd.concat([my_samp[col_name].reset_index().drop('index', axis=1), 

                        pd.DataFrame(my_emb_)], 

                       axis=1, sort=True).drop_duplicates()

    my_embed.columns=[col_name]+[col_name+'_'+e for e in map(str, range(0, my_emb_.shape[1]))]

    return my_embed
emb_lkup=[make_embed(l) for l in lo]
l_tr=[]

for i in range(0,len(lo)):

    l_tr.append(pd.merge(tr_[lo].astype('str'),emb_lkup[i], on=lo[i], how='left'))

tr_emb=pd.concat(l_tr, axis=1).drop(lo, axis=1)

tr_emb.columns=["emb"+e for e in map(str,range(0, len(tr_emb.columns)))]

tr_emb.shape
l_te=[]

for i in range(0,len(lo)):

    l_te.append(pd.merge(te_[lo].astype('str'),emb_lkup[i], on=lo[i], how='left'))

te_emb=pd.concat(l_te, axis=1).drop(lo, axis=1)

te_emb.columns=["emb"+e for e in map(str,range(0, len(te_emb.columns)))]

te_emb.shape
%%markdown 

# Engineered features
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

tr_['Transaction_day_of_week'] = np.floor((tr_['TransactionDT'] / (3600 * 24) - 1) % 7)

te_['Transaction_day_of_week'] = np.floor((te_['TransactionDT'] / (3600 * 24) - 1) % 7)

tr_['Transaction_hour'] = np.floor(tr_['TransactionDT'] / 3600) % 24

te_['Transaction_hour'] = np.floor(te_['TransactionDT'] / 3600) % 24
%%markdown

# Create train and test
x_train=pd.concat(

    [tr_[reduce_time_vars], 

     tr_[reduce_num_vars], 

     pd.get_dummies(tr_[to_dummy], dummy_na=False), 

     tr_emb,

     tr_[['Transaction_day_of_week', 'Transaction_hour']]

    ], 

    axis=1).drop('TransactionDT', axis=1)



x_test=pd.concat(

    [te_[reduce_time_vars], 

     te_[reduce_num_vars], 

     pd.get_dummies(te_[to_dummy], dummy_na=False), 

     te_emb,

     te_[['Transaction_day_of_week', 'Transaction_hour']]

    ], 

    axis=1).drop('TransactionDT', axis=1)



x_train.shape, x_test.shape, np.unique(tr_.isFraud, return_counts=True)
%%markdown

# Parameter grid
# model = xgboost.XGBClassifier(objective = 'binary:logistic')



# n_iter = 100



# param_dist = {

#  "subsample"        : [0.20, 0.50, 0.80],

#  "learning_rate"    : [0.025, 0.05] ,

#  "max_depth"        : [5],

#  "eval_metric"      : ["auc"],

#  "min_child_weight" : sp_randint(1, 11),

#  "colsample_bytree" : [0.1, 0.3, 0.5, 0.7],

#  "n_estimators"     : [50],

#  "reg_lambda"       : st.uniform(0.0, 1.0),

#  "verbosity"        : [3],

#  "scale_pos_weight" : [0.25, 0.5, 0.75, 1]

# }



# clf = RandomizedSearchCV(model,

#                          param_dist, 

#                          verbose=3, 

#                          n_jobs=3,

#                          cv=3,

#                          n_iter=n_iter,

#                          scoring = 'roc_auc'

#                   )



# clf.fit(x_train,tr_.isFraud)

# clf.best_params_
my_param_dist = { 

 "learning_rate"    : 0.25,

 "max_depth"        : 5,   

 "n_estimators"     : 500,

 'reg_lambda'       : 0.8492390698513373,

 'min_child_weight' : 10,

 'colsample_bytree' : 0.7,

 'subsample'        : 0.5,

 'scale_pos_weight' : 1,

 'eval_metric'      : 'auc',

 'objective'        : 'binary:logistic'

}
X_train, X_test, Y_train, Y_test = train_test_split(x_train, tr_.isFraud, test_size=0.001)
%%markdown

# Fit model
watchlist = [(xgboost.DMatrix(X_train, Y_train), 'train'), (xgboost.DMatrix(X_test, Y_test), 'valid')]

bst_xgb=xgboost.train(my_param_dist, xgboost.DMatrix(X_train, Y_train), num_boost_round=100, early_stopping_rounds=100, evals=watchlist)
h_preds_=pd.DataFrame(bst_xgb.predict(xgboost.DMatrix(X_test, label=None),ntree_limit=bst_xgb.best_ntree_limit), columns=['isFraud'])
%%markdown

# Small holdout
(

print(' AUC :      {0}\n Accuracy : {1}\n Confusion Matrix :\n {2}'

      .format(roc_auc_score(Y_test, h_preds_.isFraud), 

              accuracy_score(Y_test, np.where(h_preds_.isFraud > 0.5, 1,0)),

              confusion_matrix(Y_test, np.where(h_preds_.isFraud > 0.5, 1,0))))

)
%%markdown

# Write out results and pickled model
ms_=x_train.columns.difference(x_test.columns)

x_space=pd.DataFrame(np.zeros(shape=(len(x_test), len(ms_))),columns=ms_)



out=pd.concat([pd.DataFrame(te_.TransactionID), pd.DataFrame(bst_xgb.predict(xgboost.DMatrix(pd.concat([x_space, x_test], axis=1)[x_train.columns], label=None)), columns=['isFraud'])], axis=1)

out.to_csv('preds.csv', index=False)