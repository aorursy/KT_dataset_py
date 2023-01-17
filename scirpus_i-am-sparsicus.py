import gc

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score

from scipy.sparse import csc_matrix, hstack, csr_matrix

from scipy.sparse.linalg import lsqr

from sklearn.preprocessing import OneHotEncoder

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
alldata = pd.concat([train_df[train_df.columns[2:]],test_df[train_df.columns[2:]]])

ohe = OneHotEncoder()

cuts = 40

sparsedata = None

for c in alldata.columns:

    if(sparsedata is None):

        sparsedata = ohe.fit_transform(pd.cut(alldata[c],cuts,labels=False).values.reshape(-1,1))

    else:

        sparsedata = hstack([sparsedata,ohe.fit_transform(pd.cut(alldata[c],cuts,labels=False).values.reshape(-1,1))])

target = train_df['target'].values

sparsedata = csr_matrix(sparsedata,dtype=float)
noOfFolds = 5

skf = StratifiedKFold(n_splits=noOfFolds, shuffle=True, random_state=42)

oof = np.zeros(len(target))

predictions = np.zeros(sparsedata.shape[0]-len(target))



for fold, (trn_idx, val_idx) in enumerate(skf.split(sparsedata[:len(target)], target)):

    X_train, y_train = sparsedata[:len(target)][trn_idx], target[trn_idx]

    X_valid, y_valid = sparsedata[:len(target)][val_idx], target[val_idx]

    x = lsqr(X_train, y_train,damp=1.0)[0]

    trpreds = X_valid.dot(x)  

    tepreds = sparsedata[len(target):].dot(x)  

    oof[val_idx] = trpreds

    predictions += tepreds/noOfFolds

    val_score = roc_auc_score(y_valid, trpreds)

    print(fold,val_score)

    del X_train

    del y_train 

    del X_valid

    del y_valid

    gc.collect()

roc_auc_score(target, oof)
x = lsqr(sparsedata[:len(target)], target,damp=1.0)[0]

f = pd.DataFrame()

f['ID_code'] = test_df.ID_code.values

f['target'] = sparsedata[len(target):].dot(x)  

f.to_csv('straightlinalg.csv',index=False)