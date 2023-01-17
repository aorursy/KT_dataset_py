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
data = pd.concat([

       pd.read_csv("../input/pageviews/pageviews.csv", parse_dates=["FEC_EVENT"]),

       pd.read_csv("../input/pageviews_complemento/pageviews_complemento.csv", parse_dates=["FEC_EVENT"])

])

data
X_test = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_test.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_test = pd.concat(X_test, axis=1)
data = data[data.FEC_EVENT.dt.month < 10]

X_train = []

for c in data.drop(["USER_ID", "FEC_EVENT"], axis=1).columns:

    print("haciendo", c)

    temp = pd.crosstab(data.USER_ID, data[c])

    temp.columns = [c + "_" + str(v) for v in temp.columns]

    X_train.append(temp.apply(lambda x: x / x.sum(), axis=1))

X_train = pd.concat(X_train, axis=1)
features = list(set(X_train.columns).intersection(set(X_test.columns)))

X_train = X_train[features]

X_test = X_test[features]
y_prev = pd.read_csv("../input/conversiones/conversiones.csv")

y_train = pd.Series(0, index=X_train.index)

idx = set(y_prev[y_prev.mes >= 10].USER_ID.unique()).intersection(

        set(X_train.index))

y_train.loc[list(idx)] = 1
from lightgbm import LGBMClassifier

from sklearn import model_selection

from sklearn.metrics import roc_auc_score



fi = []

test_probs = []

i = 0

for train_idx, valid_idx in model_selection.KFold(n_splits=10, shuffle=True).split(X_train):

    i += 1

    Xt = X_train.iloc[train_idx]

    yt = y_train.loc[X_train.index].iloc[train_idx]



    Xv = X_train.iloc[valid_idx]

    yv = y_train.loc[X_train.index].iloc[valid_idx]



    learner = LGBMClassifier(n_estimators=10000)

    learner.fit(Xt, yt,  early_stopping_rounds=10, eval_metric="auc",

                eval_set=[(Xt, yt), (Xv, yv)])

    

    test_probs.append(pd.Series(learner.predict_proba(X_test)[:, -1],

                                index=X_test.index, name="fold_" + str(i)))

    fi.append(pd.Series(learner.feature_importances_ / learner.feature_importances_.sum(), index=Xt.columns))



test_probs = pd.concat(test_probs, axis=1).mean(axis=1)

test_probs.index.name="USER_ID"

test_probs.name="SCORE"

test_probs.to_csv("benchmark.zip", header=True, compression="zip")

fi = pd.concat(fi, axis=1).mean(axis=1)
os.listdir("./")