# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from keras.layers import Dense, Activation, Dropout

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from contextlib import contextmanager

from pandas.core.common import SettingWithCopyWarning

from sklearn.model_selection import KFold

import gc

import xgboost as xgb

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import time

import warnings



from contextlib import contextmanager

from pandas.core.common import SettingWithCopyWarning

from sklearn.model_selection import KFold





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
X_train = train_data.drop(['target', 'ID_code'], axis=1)

X_test = test_data.drop(['ID_code'], axis=1)
sc = StandardScaler()

std = sc.fit_transform(X_test + X_train)
X_train_std = sc.fit_transform(X_train)

X_test_std = sc.fit_transform(X_test)
Y_train = train_data[['target']]
X_train_new ,X_val ,Y_train_new , Y_val = train_test_split(X_train_std,Y_train,test_size=0.33,random_state=44)
print(X_train_new.shape)

print(Y_train_new.shape)
model = Sequential()

model.add(Dense(512,activation = 'relu', input_dim = 200))

model.add(Dropout(0.45))

model.add(Dense(64,activation='relu'))



model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')



model.fit(X_train_new,Y_train_new,validation_data = (X_val,Y_val),epochs=20,batch_size=256)
y_preds_1 = model.predict(X_test_std)

for i in range(len(y_preds_1)):

    if y_preds_1[i]<0.5:

        y_preds_1[i]=0

    else:

        y_preds_1[i]=1
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



target = train['target']

train_df = train.drop(['ID_code', 'target'],axis = 1)

test_df  = test.drop(['ID_code'],axis = 1)

feats = train_df.columns



fold_xgb = KFold(n_splits=5, shuffle=False, random_state=114)

def kfold_xgboost(train_df, test_df, debug = False):

    oof_preds_xgb = np.zeros(train_df.shape[0])

    sub_preds_xgb = np.zeros(test_df.shape[0])

    xgb_params = {

    'objective': 'binary:logitraw',

    'eval_metric': 'auc',

    'booster': 'gbtree',

    'n_jobs': 4,

    'tree_method': 'hist',

    'eta': 0.2,

    'grow_policy': 'lossguide',

    'max_delta_step': 2,

    'seed': 538,

    'colsample_bylevel': 0.9,

    'colsample_bytree': 0.8,

    'gamma': 1.0,

    'learning_rate': 0.1,

    'max_bin': 64,

    'max_depth': 8,

    'max_leaves': 15,

    'min_child_weight': 16,

    'reg_alpha': 1e-06,

    'reg_lambda': 1.0,

    'subsample': 0.7}

    for fold_, (train_idx, valid_idx) in enumerate(fold_xgb.split(train_df.values)):

        train_x, train_y = train_df.iloc[train_idx], train['target'].iloc[train_idx]

        valid_x, valid_y = train_df.iloc[valid_idx], train['target'].iloc[valid_idx]

        print("fold n °{}".format(fold_))

        trn_Data = xgb.DMatrix(train_x, label = train_y)

        val_Data = xgb.DMatrix(valid_x, label = valid_y)

        watchlist = [(trn_Data, "Train"), (val_Data, "Valid")]

        print("xgb" + str(fold_) + "-" * 50)

        num_rounds = 10000

        xgb_model = xgb.train(xgb_params, trn_Data,num_rounds,watchlist,early_stopping_rounds=50, verbose_eval= 1000)

        oof_preds_xgb[valid_idx] = xgb_model.predict(xgb.DMatrix(train_df.iloc[valid_idx][feats]), ntree_limit = xgb_model.best_ntree_limit + 50)

        sub_preds_xgb = xgb_model.predict(xgb.DMatrix(test_df[feats]),ntree_limit= xgb_model.best_ntree_limit)/fold_xgb.n_splits

        

        del train_idx,valid_idx

        gc.collect()

    xgb.plot_importance(xgb_model)

    plt.savefig("importance.png")

    return sub_preds_xgb

Preds_xgb = kfold_xgboost(train_df, test_df, debug = False)
np.save('Preds_xgb.npy',Preds_xgb)





for i in range(len(preds_final)):

    if preds_final[i]>0.5:

        preds_final[i]=1

    else:

        preds_final[i]=0

        

submission['target'] = (Preds_xgb + y_preds_1)/2

submission.to_csv(submission.csv, index = False)