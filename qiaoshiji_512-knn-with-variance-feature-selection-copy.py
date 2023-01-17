# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



###############################################################################

################################## Imports

###############################################################################

import numpy as np

import pandas as pd

import os

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import time

import gc

from sklearn import neighbors

from sklearn import metrics, preprocessing

from sklearn.feature_selection import VarianceThreshold



###############################################################################

################################## Data

###############################################################################

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



X = train.iloc[:,1:257]

X_test = test.iloc[:,1:257]

Y = train.iloc[:,257]



cols = [c for c in train.columns if c not in ['id', 'target']]



cols.remove('wheezy-copper-turtle-magic')



prediction = np.zeros(len(test))



scaler = preprocessing.StandardScaler()

scaler.fit(pd.concat([X, X_test]))

scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)



###############################################################################

################################## Model

###############################################################################

skf = StratifiedKFold(n_splits=5, random_state=42)



oof = np.zeros(len(train))

st = time.time()

for i in range(512):

    if i%5==0: print('Model : ',i, 'Time : ', time.time()-st)



    x = train[train['wheezy-copper-turtle-magic']==i]

    x_test = test[test['wheezy-copper-turtle-magic']==i]

    y = Y[train['wheezy-copper-turtle-magic']==i]

    idx = x.index

    idx_test = x_test.index

    x.reset_index(drop=True,inplace=True)

    x_test.reset_index(drop=True,inplace=True)

    y.reset_index(drop=True,inplace=True)

    

    clf = lgb.LGBMRegressor()

    clf.fit(x[cols],y)

    important_features = [i for i in range(len(cols)) if clf.feature_importances_[i] > 0] 

    cols_important = [cols[i] for i in important_features]

    

    sel = VarianceThreshold(threshold=1.5).fit(x[cols])

    train3 = sel.transform(x[cols])

    test3 = sel.transform(x_test[cols])

    

    skf = StratifiedKFold(n_splits=10, random_state=42)

    for train_index, valid_index in skf.split(train3, y):

        # KNN

        clf = neighbors.KNeighborsClassifier(5)

        clf.fit(train3[train_index], y[train_index])

        oof[idx[valid_index]] = clf.predict_proba(train3[valid_index])[:,1]

        prediction[idx_test] += clf.predict_proba(test3)[:,1] / 25.0

    print(i, 'oof auc : ', roc_auc_score(Y[idx], oof[idx]))

        

print('total auc : ',roc_auc_score(train['target'],oof))



sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = prediction

sub.to_csv('submission.csv',index=False)