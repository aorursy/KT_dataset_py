import gc

import os

import logging

from datetime import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

from sklearn import decomposition

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# Code from: https://stackoverflow.com/questions/45028260/gaussian-to-uniform-distribution-conversion-has-errors-at-the-edges-of-uniform-d

def gaussian_estimation(vector):

    mu = np.mean(vector)

    sig = np.std(vector)

    return mu, sig



# Adjusts the data so it forms a gaussian with mean of 0 and std of 1

def gaussian_normalization(vector, char = None):

    if char is None:

        mu , sig = gaussian_estimation(vector)

    else:

        mu = char[0]

        sig = char[1]

    normalized = (vector-mu)/sig

    return normalized



# Taken from https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function

def CDF(x, max_i = 100):

    sum = x

    value = x

    for i in np.arange(max_i)+1:

        value = value*x*x/(2.0*i+1)

        sum = sum + value

    return 0.5 + (sum/np.sqrt(2*np.pi))*np.exp(-1*(x*x)/2)



def gaussian_to_uniform(vector, if_normal = False):

    if (if_normal == False):

        vector = gaussian_normalization(vector)

    uni = np.apply_along_axis(CDF, 0, vector)

    return uni

# Original DB

train_df_orig = pd.read_csv("../input/train.csv")

test_df_orig = pd.read_csv("../input/test.csv")

target = train_df_orig['target']

orig_features = [c for c in train_df_orig.columns if c not in ['ID_code', 'target']]
# Convert features to Uniform Distribution

train_uniform = pd.DataFrame()

test_uniform = pd.DataFrame()

bin_cnt = 10**3

for namecol in tqdm_notebook(orig_features):

    train_uniform[namecol+'_unif'] = gaussian_to_uniform(train_df_orig[namecol])

    test_uniform[namecol+'_unif'] = gaussian_to_uniform(test_df_orig[namecol]) 
# Code thanks to: https://www.kaggle.com/youhanlee/yh-eda-i-want-to-see-all

from scipy.stats import ks_2samp

target_mask = train_df_orig['target'] == 1

non_target_mask = train_df_orig['target'] == 0 

statistics_array = []

for col in tqdm_notebook(train_uniform.columns):

    statistic, pvalue = ks_2samp(train_uniform.loc[non_target_mask, col], train_uniform.loc[target_mask, col])

    statistics_array.append(statistic)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.kdeplot(train_uniform.loc[non_target_mask, col], ax=ax, label='Target == 0')

    sns.kdeplot(train_uniform.loc[target_mask, col], ax=ax, label='Target == 1')

    ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))

    plt.show()

    

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    sns.distplot(train_uniform.loc[non_target_mask, col], bins=50, kde=False, rug=False);

    sns.distplot(train_uniform.loc[target_mask, col], bins=50, kde=False, rug=False);

    plt.show()
train_df = pd.concat([train_df_orig[orig_features],train_uniform],axis=1)

test_df = pd.concat([test_df_orig[orig_features],test_uniform],axis=1)

features = [c for c in train_df.columns if c not in ['target', 'ID_code']]



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



print("CV score:{:<8.5f} CV_stats:[{:<8.5f}, {:<8.5f} ({:<8.5f}), {:<8.5f}]".format(roc_auc_score(target, oof),np.min(AUC_l),np.mean(AUC_l), np.std(AUC_l), np.max(AUC_l)))

score = roc_auc_score(target, oof)

best_iter = np.mean(iter_l)

best_score_final = score
# Submission

sub_df = pd.DataFrame({"ID_code":test_df_orig["ID_code"].values})

sub_df["target"] = predictions

sub_df.to_csv('submission.csv.gz', index=False, compression='gzip')

print(pd.read_csv('submission.csv.gz').head())