import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



from tqdm import tqdm

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



scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)



oof = np.zeros(len(train))

st = time.time()

for i in tqdm(range(512)):

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

    test3 = sel.transform(x_test[cols]) #on peut mettre cols_important aussi

    n_folds=10

    skf = StratifiedKFold(n_splits=n_folds, random_state=42)

    for train_index, valid_index in skf.split(train3, y):

        # KNN

#         clf = neighbors.KNeighborsClassifier(n_neighbors  =7, p=2, weights ='distance')

#         clf.fit(train3[train_index], y[train_index])

#         oof[idx[valid_index]] = clf.predict_proba(train3[valid_index])[:,1]

#         prediction[idx_test] += clf.predict_proba(test3)[:,1] / 25.0

            param = {

            'n_jobs' : -1,

            'boosting': 'gbdt',

            'learning_rate': 0.05,

            #'max_depth': 24,

            'metric': 'auc',

            #'num_leaves': 454,

            'objective': 'binary',

            #'subsample': 0.94

            }



            train_dataset = lgb.Dataset(train3[train_index], y[train_index])

            val_dataset = lgb.Dataset(train3[valid_index], y[valid_index])

            

            clf = lgb.train(param, train_dataset, valid_sets=[train_dataset, val_dataset], verbose_eval=False,

                              num_boost_round=5000, early_stopping_rounds=250)

            

            oof[idx[valid_index]] = clf.predict(train3[valid_index], num_iteration=clf.best_iteration)

            prediction[idx_test] += clf.predict(test3, num_iteration=clf.best_iteration) / n_folds

            

    print(i, 'oof auc : ', roc_auc_score(Y[idx], oof[idx]))

        

print('total auc : ',roc_auc_score(train['target'],oof))



sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = prediction

sub.to_csv('submission.csv',index=False)