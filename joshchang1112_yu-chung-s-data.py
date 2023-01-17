# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import numpy as np

import pandas as pd

from sklearn import preprocessing

import xgboost as xgb

import scipy as sp

from scipy import stats

import matplotlib.pyplot as plt
data = pd.read_csv("../input/.csv")
data
data['Label'] = data['Label'].map(dict(zip(['Non-Converter', 'Converter'], [0, 1])))

y_train = data['Label']

x_train = data.copy()

# x_train = x_train.drop(, axis=1)

# x_train = x_train.drop(, axis=1)

x_train = x_train.drop(['Study ID', 'Name', 'Label'], axis=1)



# 先挑選年齡和性別

x = x_train.iloc[:, :2]
column_num = []

for i in range(2, 31):

    column_num.append(i)





import xgboost as xgb

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, accuracy_score



EPOCHS = 5



# 從頭開始 挑選20個特徵

for i in range(20):

    print("ROUND" + str(i + 1))

    print("===================================")

    avg = 0

    best_avg = 0    

    best_id = -1

    for j in column_num:

        avg = 0

        x_tmp = pd.concat([x, x_train.iloc[:, j]], axis=1)

        kf = StratifiedKFold(n_splits=EPOCHS, random_state=208, shuffle=True)

        for tr_idx, val_idx in kf.split(x_tmp, y_train):

            clf = xgb.XGBClassifier(

                n_estimators=500,

                learning_rate=0.5,

                tree_method='gpu_hist',

                seed=1000

            )



            X_tr, X_vl = x_tmp.iloc[tr_idx, :], x_tmp.iloc[val_idx, :]

            y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]



            clf.fit(X_tr, y_tr, eval_metric="auc", 

                    eval_set=[(X_vl, y_vl)], 

                    verbose=False, 

                    early_stopping_rounds = 200)



            predict = clf.predict(X_vl)

            acc = accuracy_score(y_vl, predict)

            #print(acc)

            avg += acc / EPOCHS



        print("AVG Accracy: " + str(avg))

        if avg > best_avg:

            print("BEST!!!")

            print(avg)

            best_avg = avg

            best_id = j

    

    x = pd.concat([x, x_train.iloc[:, best_id]], axis=1)

    column = column_num.remove(best_id)

    print("===================================")

    

# 全部特徵

avg = 0

for tr_idx, val_idx in kf.split(x_train, y_train):

    clf = xgb.XGBClassifier(

        n_estimators=500,

        learning_rate=0.5,

        tree_method='gpu_hist',

        seed=1000

    )



    X_tr, X_vl = x_train.iloc[tr_idx, :], x_train.iloc[val_idx, :]

    y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]



    clf.fit(X_tr, y_tr, eval_metric="auc", 

            eval_set=[(X_vl, y_vl)], 

            verbose=False, 

            early_stopping_rounds = 200)



    predict = clf.predict(X_vl)

    acc = accuracy_score(y_vl, predict)

    print(acc)

    avg += acc / EPOCHS



print("AVG Accracy: " + str(avg))
feature_important = clf.get_booster().get_score(importance_type="weight")

# print(len(feature_important))

# print(feature_important)

keys = list(feature_important.keys())

values = list(feature_important.values())

data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)

feature_select = data.index[:120]

data.head(60)