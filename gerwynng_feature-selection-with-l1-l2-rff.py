import pandas as pd

import seaborn as sns

import numpy as np

import lightgbm as lgbm

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

import gc
def reduce_mem_usage(df):

    for col in df.columns:

        if df[col].dtype=='float64': df[col] = df[col].astype('float32')

        if df[col].dtype=='int64': df[col] = df[col].astype('int32')

    return df
# import data

train_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID'))

train_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID'))



# merge

train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)



# Prepare data

X = train_df.drop(['TransactionDT','isFraud'],axis=1)

y = train_df['isFraud']



for col in X.columns:

    if X[col].dtype == 'object':

        le = LabelEncoder()

        X[col] = le.fit_transform(list(X[col].astype(str).values))


lgbm_params = {'num_leaves':165,

               'min_child_weight': 10,

               'min_child_samples': 10,

               'min_split_gain': 0,

               'subsample': .632,

               'subsample_freq':1,

               'objective': 'binary',

               'max_depth': -1,

               'learning_rate': 0.1, 

               "boosting_type": "rf",

               "bagging_seed": 11,

               "metric": 'auc',

               'random_state': 47,

               'num_rounds': 400,

               'reg_alpha':10 # Tweak Me

              }
### train/test/split

train_idx = int(len(X)*0.6)

x_trn = X[:train_idx]

y_trn = y[:train_idx]



### hold out valid

idx = int(0.8*len(X))



y_ho = y[idx:]

x_ho = X[idx:]
# CV

NFOLDS = 5

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=47)



columns = x_trn.columns

splits = folds.split(x_trn, y_trn)

y_preds = np.zeros(x_ho.shape[0])



l1_feature_importances = pd.DataFrame()

l1_feature_importances['feature'] = columns
for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = x_trn[columns].iloc[train_index], x_trn[columns].iloc[valid_index]

    y_train, y_valid = y_trn.iloc[train_index], y_trn.iloc[valid_index]

    

    dtrain = lgbm.Dataset(X_train, label=y_train)

    dvalid = lgbm.Dataset(x_ho, label=y_ho)



    clf = lgbm.train(lgbm_params, dtrain, valid_sets = [dtrain, dvalid], 

                    verbose_eval=200)

    

    l1_feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(x_ho)



    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

del clf, x_trn, y_trn, x_ho, y_ho
l1_feature_importances['average'] = l1_feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

l1_feature_importances.to_csv('l1_feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=l1_feature_importances.sort_values(by='average', ascending=False).head(30), x='average', y='feature');
print('L1 Zero Importance Features:')

print(l1_feature_importances[l1_feature_importances['average'] == 0]['feature'])
lgbm_params = {'num_leaves':165,

               'min_child_weight': 10,

               'min_child_samples': 10,

               'min_split_gain': 0,

               'subsample': .632,

               'subsample_freq':1,

               'objective': 'binary',

               'max_depth': -1,

               'learning_rate': 0.1, 

               "boosting_type": "rf",

               "bagging_seed": 11,

               "metric": 'auc',

               'random_state': 47,

               'num_rounds': 400,

               'reg_lambda':5 # Tweak Me

              }
### train/test/split

train_idx = int(len(X)*0.6)

x_trn = X[:train_idx]

y_trn = y[:train_idx]



### hold out valid

idx = int(0.8*len(X))



y_ho = y[idx:]

x_ho = X[idx:]



# CV

NFOLDS = 5

folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=47)



columns = x_trn.columns

splits = folds.split(x_trn, y_trn)

y_preds = np.zeros(x_ho.shape[0])



l2_feature_importances = pd.DataFrame()

l2_feature_importances['feature'] = columns
for fold_n, (train_index, valid_index) in enumerate(splits):

    X_train, X_valid = x_trn[columns].iloc[train_index], x_trn[columns].iloc[valid_index]

    y_train, y_valid = y_trn.iloc[train_index], y_trn.iloc[valid_index]

    

    dtrain = lgbm.Dataset(X_train, label=y_train)

    dvalid = lgbm.Dataset(x_ho, label=y_ho)



    clf = lgbm.train(lgbm_params, dtrain, valid_sets = [dtrain, dvalid], 

                    verbose_eval=200)

    

    l2_feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()

    

    y_pred_valid = clf.predict(x_ho)



    del X_train, X_valid, y_train, y_valid

    gc.collect()

    

del clf, x_trn, y_trn, x_ho, y_ho
l2_feature_importances['average'] = l2_feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

l2_feature_importances.to_csv('l2_feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=l2_feature_importances.sort_values(by='average', ascending=False).head(30), x='average', y='feature');
print('L2 Zero Importance Features:')

print(l2_feature_importances[l2_feature_importances['average'] == 0]['feature'])
### Reference

# https://www.kaggle.com/ogrellier/lgbm-regularized-random-forest