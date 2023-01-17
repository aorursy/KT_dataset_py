import gc

import os

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold
# Original DB

train_df_orig = pd.read_csv("../input/train.csv")

test_df_orig = pd.read_csv("../input/test.csv")

target = train_df_orig['target']

orig_features = [c for c in train_df_orig.columns if c not in ['ID_code', 'target']]



# DF TOTAL 

df_total = pd.concat([train_df_orig[orig_features], test_df_orig[orig_features]], axis=0)

df_total = df_total.reset_index(drop=True)



# DF sorted by 'var_68'

df_total['target'] = np.concatenate([train_df_orig.target.values,np.full(len(test_df_orig),np.nan)])

sorted_total_df = df_total.sort_values('var_68', ascending=False).copy()

sorted_total_df.head()
size_fil = 8

size_fil_med = int(size_fil/2)

target_s_filter = sorted_total_df.target.reset_index(drop=True).values

target_s_filter_end = target_s_filter.copy()

for i in np.arange(size_fil_med,len(target_s_filter)-size_fil_med):

    vec = np.concatenate([target_s_filter[(i-size_fil_med):i],target_s_filter[(i+1):(i+1+size_fil_med)]])

    vec = vec[~np.isnan(vec)]

    if len(vec)==0:

        target_s_filter_end[i] = 0.0

    else:

        target_s_filter_end[i] = np.mean(vec)

target_s_filter_end[np.isnan(target_s_filter_end)] = 0.0
sns.set(rc={'figure.figsize':(24,12)})

line=sns.lineplot(data=pd.DataFrame({'MAF of Target':target_s_filter_end[16000:16400],

                                     'Target':np.array(target_s_filter)[16000:16400]}))
# Include new feature

sorted_total_df['target_filter'] = target_s_filter_end

total_df_orig_reorder = sorted_total_df.sort_index()

train_df = total_df_orig_reorder.iloc[:len(train_df_orig)]

test_df = total_df_orig_reorder.iloc[len(train_df_orig):]

train_df.head()
# Remove var_68

features = [c for c in train_df.columns if c not in ['target', 'var_68']]



param = {

    'bagging_freq': 5,          

    'bagging_fraction': 0.30,   

    'boost_from_average':'false',   

    'boost': 'gbdt',

    'feature_fraction': 0.03368,   

    'learning_rate': 0.01,      

    'max_depth': -1,                

    'metric':'auc',

    'min_data_in_leaf': 80,     

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 4,           

    'tree_learner': 'serial',   

    'objective': 'binary',      

    'verbosity': 1

}



nfolds = 5



folds = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=31415)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

AUC_l = []

iter_l = []

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])



    num_round = 500000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, 

                    early_stopping_rounds = 250)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)



    AUC_l.append(roc_auc_score(target[val_idx], oof[val_idx]))

    iter_l.append(clf.best_iteration)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits



print("Width Filter {} CV score:{:<8.5f} CV_stats:[{:<8.5f}, {:<8.5f} ({:<8.5f}), {:<8.5f}]".format(size_fil,roc_auc_score(target, oof),np.min(AUC_l),np.mean(AUC_l), np.std(AUC_l), np.max(AUC_l)))

score = roc_auc_score(target, oof)

best_iter = np.mean(iter_l)

best_score_final = score
# Submission

sub_df = pd.DataFrame({"ID_code":test_df_orig["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv('submission.csv.gz', index=False, compression='gzip')

print(pd.read_csv('submission.csv.gz').head())