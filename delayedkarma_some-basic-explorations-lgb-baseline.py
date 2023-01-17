import datetime
import gc
import numpy as np
import os
import pandas as pd
import random
import nltk
import string

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import skew, kurtosis

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb

from tqdm import tqdm
df_train = pd.read_csv('../input/20-newsgroups-ciphertext-challenge/train.csv')
df_train.head()
df_test = pd.read_csv('../input/20-newsgroups-ciphertext-challenge/test.csv')
df_test.head()
news_list = pd.read_csv('../input/20-newsgroups/list.csv')
news_list.shape
print(df_train.shape, df_test.shape)
len(df_train['target'].value_counts()) # 20 Newsgroups -- checks out
# Are the classes balanced?

count_target = df_train['target'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(count_target.index, count_target.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Target', fontsize=12);
df_train['ciphertext'].iloc[0]
df_train.info()
## Basic features (a lot of these ideas from https://www.kaggle.com/opanichev/lightgbm-and-simple-features)

def add_feats(df): # Some of these features might be strongly correlated
    
    tqdm.pandas('add_basic')
    df['len'] = df['ciphertext'].progress_apply(lambda x: len(str(x))) # Characters
    df['unique'] = df['ciphertext'].progress_apply(lambda x: len(set(str(x)))) # Unique characters
    df['punctuations'] = df['ciphertext'].progress_apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df['uniq_punctuations'] = df['ciphertext'].progress_apply(lambda x: len(set([c for c in str(x) if c in string.punctuation])))
    df['letters'] = df['ciphertext'].progress_apply(lambda x: len([c for c in str(x) if c.isalpha()]))
    df['uniq_letters'] = df['ciphertext'].progress_apply(lambda x: len(set([c for c in str(x) if c.isalpha()])))
    df['numbers'] = df['ciphertext'].progress_apply(lambda x: len([c for c in str(x) if c.isdigit()]))
    df['uniq_numbers'] = df['ciphertext'].progress_apply(lambda x: len(set([c for c in str(x) if c.isdigit()])))
    df['uppercase'] = df['ciphertext'].progress_apply(lambda x: len([c for c in str(x) if c.isupper()]))
    df['uniq_uppercase'] = df['ciphertext'].progress_apply(lambda x: len(set([c for c in str(x) if c.isupper()])))
    df['lowercase'] = df['ciphertext'].progress_apply(lambda x: len([c for c in str(x) if c.islower()]))
    df['uniq_lowercase'] = df['ciphertext'].progress_apply(lambda x: len(set([c for c in str(x) if c.islower()])))
add_feats(df_train)
add_feats(df_test)
df_train.head()
plt.figure(figsize=(12,12))
sns.violinplot(x='target', y='unique', data=df_train)
plt.xlabel('Target', fontsize=12)
plt.ylabel('Number of unique characters in text', fontsize=12)
plt.title("Number of unique characters by target", fontsize=15);
plt.figure(figsize=(12,12))
sns.violinplot(x='target', y='uniq_punctuations', data=df_train)
plt.xlabel('Target', fontsize=12)
plt.ylabel('Number of unique punctuations in text', fontsize=12)
plt.title("Number of unique punctuations by target", fontsize=15);
fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(20,20))
sns.violinplot(x='difficulty', y='unique', data=df_train, ax=ax1)
ax1.set_xlabel('Difficulty', fontsize=12)
ax1.set_ylabel('Number of unique characters in text', fontsize=12)
sns.violinplot(x='difficulty', y='uniq_punctuations', data=df_train, ax=ax2)
ax2.set_xlabel('Difficulty', fontsize=12)
ax2.set_ylabel('Number of unique punctuations in text', fontsize=12)
sns.violinplot(x='difficulty', y='numbers', data=df_train, ax=ax3)
ax3.set_xlabel('Difficulty', fontsize=12)
ax3.set_ylabel('Number of numbers in text', fontsize=12)
sns.violinplot(x='difficulty', y='uppercase', data=df_train, ax=ax4)
ax4.set_xlabel('Difficulty', fontsize=12)
ax4.set_ylabel('Number of uppercase in text', fontsize=12);
df_train.corr()['target'] # Some of these features seem to have strong negative correlations with the target
## Unique punctuations matter apparently
cols_to_drop = ['Id','ciphertext']
X = df_train.drop(cols_to_drop, axis=1, errors='ignore')

feature_names = list(X.columns)

y = df_train['target'].values
X = X.values

X_test = df_test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = df_test['Id'].values
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': -1,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0,
    'num_class': df_train['target'].nunique()
}
cnt = 0
p_buf = []
p_valid_buf = []
n_splits = 5
kf = KFold(
    n_splits=n_splits, 
    random_state=0)
err_buf = []   
undersampling = 0
print(X.shape, y.shape)
print(X_test.shape)
n_features = X.shape[1]

for train_index, valid_index in kf.split(X, y):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = lgb_params.copy() 

    lgb_train = lgb.Dataset(
        X[train_index], 
        y[train_index], 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X[valid_index], 
        y[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(20):
            if i < len(tuples):
                print(tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X[valid_index], num_iteration=model.best_iteration)
    err = f1_score(y[valid_index], np.argmax(p, axis=1), average='macro')

    print('{} F1: {}'.format(cnt + 1, err))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    err_buf.append(err)

    cnt += 1

    del model, lgb_train, lgb_valid, p
    gc.collect

    # Train on one fold
#     if cnt > 0:
#         break


err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.6f} +/- {:.6f}'.format(err_mean, err_std))

preds = p_buf/cnt
subm = pd.DataFrame()
subm['Id'] = id_test
subm['Predicted'] = np.argmax(preds, axis=1)
subm.to_csv('submission.csv', index=False)
