import gc

import os



import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import roc_auc_score
train_features_path = '../input/baseline-features/train_features.csv'

test_features_path = '../input/baseline-features/test_features.csv'



train = pd.read_csv(train_features_path)

test = pd.read_csv(test_features_path)
train = train.sort_values('TransactionDT')

test = test.sort_values('TransactionDT')
import matplotlib.pyplot as plt



fig, axs = plt.subplots(1,2, figsize=(16,4))

train.groupby(['TransactionDT'])['TransactionDT'].size().plot(ax=axs[0])

test.groupby(['TransactionDT'])['TransactionDT'].size().plot(ax=axs[1])
del test

gc.collect()
split_perc = [p*0.01 for p in range(100)]

y_means_train, y_means_valid = [],[]

for p in split_perc:

    idx = int(p*len(train))

    y_means_train.append(train['isFraud'][:idx].mean())

    y_means_valid.append(train['isFraud'][idx:].mean())
fig, ax = plt.subplots(figsize=(16,4))

ax.plot(split_perc, y_means_train, label='train')

ax.plot(split_perc, y_means_valid, label='valid')
split_perc_df = pd.DataFrame({'perc':split_perc,'train':y_means_train, 'valid':y_means_valid})

split_perc_df['diff'] = abs(split_perc_df['train']-split_perc_df['valid'])

split_perc_df.sort_values('diff').head()
0.13*len(train), 0.26*len(train)
def sample_negative_class(train, perc):

    train_pos = train[train.isFraud==1]

    train_neg = train[train.isFraud==0].sample(frac=perc)

    

    train = pd.concat([train_pos, train_neg], axis=0)

    train = train.sort_values('TransactionDT')

    return train
def fit_predict(train, valid, model_params, training_params):

    X_train = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

    y_train = train['isFraud']



    X_valid = valid.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

    y_valid = valid['isFraud']

    

    trn_data = lgb.Dataset(X_train, y_train)

    val_data = lgb.Dataset(X_valid, y_valid)



    clf = lgb.train(model_params, trn_data, 

                    training_params['num_boosting_rounds'], 

                    valid_sets = [trn_data, val_data], 

                    early_stopping_rounds = training_params['early_stopping_rounds'],

                    verbose_eval=False

                   )

    train_preds = clf.predict(X_train, num_iteration=clf.best_iteration)

    valid_preds = clf.predict(X_valid, num_iteration=clf.best_iteration)

    return train_preds, valid_preds
idx_split = int(0.74*len(train))

train_split, valid_split = train[:idx_split], train[idx_split:]
model_params = {'num_leaves': 256,

                  'min_child_samples': 79,

                  'objective': 'binary',

                  'max_depth': 15,

                  'learning_rate': 0.05,

                  "boosting_type": "gbdt",

                  "subsample_freq": 3,

                  "subsample": 0.9,

                  "bagging_seed": 11,

                  "metric": 'auc',

                  "verbosity": -1,

                  'reg_alpha': 0.3,

                  'reg_lambda': 0.3,

                  'colsample_bytree': 0.9

                 }



training_params = {'num_boosting_rounds':1000,

                   'early_stopping_rounds':100,

               }
train_sample_perc = [p*0.05 for p in range(1,20,1)]

train_scores, valid_scores = [],[]

for perc in train_sample_perc:

    print('processing for perc {}'.format(perc))

    train_sample = sample_negative_class(train_split, perc)

    train_preds, valid_preds = fit_predict(train_sample, valid_split, model_params, training_params)

    

    train_score = roc_auc_score(train_sample['isFraud'], train_preds)

    valid_score = roc_auc_score(valid_split['isFraud'], valid_preds)

    print(perc, train_score, valid_score)

    train_scores.append(train_score)

    valid_scores.append(valid_score)
fig, axs = plt.subplots(2,1, figsize=(16,4))

axs[0].plot(train_sample_perc, train_scores, label='train')

axs[1].plot(train_sample_perc, valid_scores, label='valid')
sample_perc_df = pd.DataFrame({'perc':train_sample_perc,'train':train_scores, 'valid':valid_scores})

sample_perc_df.sort_values('valid', ascending=False).head()