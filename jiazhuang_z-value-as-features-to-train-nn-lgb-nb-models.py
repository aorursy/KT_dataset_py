import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import Sequential, layers, callbacks



from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression



import lightgbm as lgb
train = pd.read_csv('../input/train.csv', index_col=0)

test = pd.read_csv('../input/test.csv', index_col=0)



target = train.target.values

train.drop('target', axis=1, inplace=True)

train.shape, target.shape, test.shape, 
pos_idx = (target == 1)

neg_idx = (target == 0)

stats = []

for col in train.columns:

    stats.append([

        train.loc[pos_idx, col].mean(),

        train.loc[pos_idx, col].std(),

        train.loc[neg_idx, col].mean(),

        train.loc[neg_idx, col].std()

    ])

    

stats_df = pd.DataFrame(stats, columns=['pos_mean', 'pos_sd', 'neg_mean', 'neg_sd'])

stats_df.head()
zval1 = (train.values - stats_df.neg_mean.values) / stats_df.neg_sd.values

zval2 = (train.values - stats_df.pos_mean.values) / stats_df.pos_sd.values

tr_zval = np.column_stack([zval1, zval2])

tr_zval.shape
zval1 = (test.values - stats_df.neg_mean.values) / stats_df.neg_sd.values

zval2 = (test.values - stats_df.pos_mean.values) / stats_df.pos_sd.values

te_zval = np.column_stack([zval1, zval2])

te_zval.shape
def augment(X, y, times=8):

    # up-sample positive samples

    pos_idx = (y == 1)

    nsample = times * pos_idx.sum()

    

    X_sample = []

    for i in range(X.shape[1]):

        X_sample.append(np.random.choice(X[pos_idx, i], size=nsample))

        

    X_sample = np.column_stack(X_sample)

    

    # shuffle

    idx = np.arange(X.shape[0] + nsample)

    np.random.shuffle(idx)

    

    return np.vstack([X, X_sample])[idx], np.hstack([y, np.ones(nsample)])[idx]
nfold = 5

kfold = StratifiedKFold(n_splits=nfold, shuffle=True)



nn_oof_tr, nn_oof_te = np.zeros(tr_zval.shape[0]), 0

lgb_oof_tr, lgb_oof_te = np.zeros(tr_zval.shape[0]), 0

nb_oof_tr, nb_oof_te = np.zeros(tr_zval.shape[0]), 0



for tr_idx, va_idx in kfold.split(tr_zval, target):

    X_tr, y_tr = tr_zval[tr_idx], target[tr_idx] 

    X_va, y_va = tr_zval[va_idx], target[va_idx]

    # data augment

    X_tr, y_tr = augment(X_tr, y_tr)

    

    # NN model

    model = Sequential([

        layers.Dense(16, input_shape=(400, 1), activation='relu'),

        layers.Flatten(),

        layers.Dense(1, activation='sigmoid')

    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy'])

    earlystop = callbacks.EarlyStopping(patience=10)

    # reshape data for nn input

    X_tr_re = X_tr[:, :, np.newaxis]

    X_va_re = X_va[:, :, np.newaxis]

    model.fit(X_tr_re, y_tr, validation_data=(X_va_re, y_va), epochs=100, verbose=2, batch_size=256, callbacks=[earlystop])

    # get oof

    nn_oof_tr[va_idx] = model.predict(X_va_re).flatten()

    X_te_re = te_zval[:, :, np.newaxis]

    nn_oof_te += model.predict(X_te_re).flatten() / nfold

    

    # LGB model

    param = {

        'objective': 'binary',

        'boost': 'gbdt',

        'metric': 'auc',

        'learning_rate': 0.01,

        'num_leaves': 13,

        'max_depth': -1,

        'feature_fraction': 0.05,

        'bagging_freq': 5,

        'bagging_fraction': 0.4,

        'min_data_in_leaf': 80,

        'min_sum_hessian_in_leaf': 10,

        'num_threads': 4

    }

    trn_data = lgb.Dataset(X_tr, y_tr)

    val_data = lgb.Dataset(X_va, y_va)

    clf = lgb.train(param, trn_data, 100000, valid_sets=(val_data), early_stopping_rounds=600, verbose_eval=600)

    # get oof

    lgb_oof_tr[va_idx] = clf.predict(X_va)

    lgb_oof_te += clf.predict(te_zval) / nfold

    

    # Naive Bayes model

    bayes = GaussianNB()

    bayes.fit(X_tr, y_tr)

    nb_oof_tr[va_idx] = bayes.predict_proba(X_va)[:, 1]

    nb_oof_te += bayes.predict_proba(te_zval)[:, 1] / nfold
lr = LogisticRegression()

lr.fit(np.column_stack([nn_oof_tr, lgb_oof_tr, nb_oof_tr]), target)

pred = lr.predict_proba(np.column_stack([nn_oof_te, lgb_oof_te, nb_oof_te]))[:, 1]
pd.DataFrame({

    'ID_code': test.index,

    'target': pred

}).to_csv('sub.csv', index=False)