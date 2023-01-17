import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('../input/otto-group-product-classification-challenge/train.csv')

test = pd.read_csv('../input/otto-group-product-classification-challenge/test.csv')

sample_submit = pd.read_csv('../input/otto-group-product-classification-challenge/sampleSubmission.csv')
train['target'] = train['target'].str.replace('Class_', '')

train['target'] = train['target'].astype(int) - 1
NFOLDS = 5

RANDOM_STATE = 871972



excluded_column = ['target', 'id']

cols = [c for c in train.columns if c not in (excluded_column + [])]



folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, 

                        random_state=RANDOM_STATE)
y_pred_knn = np.zeros((len(test), 9))

oof = np.zeros((len(train), 9))

score = 0

feature_importance_df = pd.DataFrame()

valid_predict = []

for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y = train['target'])):

    print('Fold', fold_n)

    X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]

    y_train, y_valid = X_train['target'].astype(int), X_valid['target'].astype(int)

    X_train, X_valid = X_train[cols], X_valid[cols]

        

    knn = KNeighborsClassifier(n_neighbors=128)

    knn.fit(X_train, y_train)

    

    valid = knn.predict_proba(X_valid[cols])

    oof[valid_index] = valid

    score += log_loss(y_valid, valid)

    print('Fold', fold_n, 'valid loglodd', log_loss(y_valid, valid))

    

    y_pred_knn += knn.predict_proba(test[cols]) / NFOLDS

    

print('valid logloss average:', score/NFOLDS)
submit = pd.concat([sample_submit[['id']], pd.DataFrame(y_pred_knn)], axis = 1)

submit.columns = sample_submit.columns



submit.to_csv('submit.csv', index=False)
column_name = ['knn_' + str(i) for i in range(9)]

pd.DataFrame(oof, columns = column_name).to_csv('oof_knn.csv', index=False)

pd.DataFrame(y_pred_knn, columns = column_name).to_csv('submit_knn.csv', index=False)