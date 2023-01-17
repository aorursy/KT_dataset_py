# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.imputation import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

to_drop = ["job_name", "reason"]

# have to upload train by hand since for some reason it is nont available here
df_train = pd.read_csv("../input/train_data/train_data.csv").set_index("ids").drop(to_drop, axis=1)
df_train = df_train[~df_train.default.isnull()]
df_train["default"] = df_train["default"].astype("int")
df_test = pd.read_csv("../input/raladores/teste_data.csv").set_index("ids").drop(to_drop, axis=1)
print((df_train.shape, df_test.shape))

encode_cols = df_train.dtypes
encode_cols = encode_cols[encode_cols == object].index.tolist()

# stats
print(pd.concat([df_train.isnull().mean(), df_train.dtypes, df_train.T.apply(lambda x: x.nunique(), axis=1)], axis=1))

def get_encoder(df, col):
    dft = df[col].astype(str).to_frame().copy()
    dft["count"] = 1
    return dft.groupby(col).count().to_dict()["count"]
    
def encode_all(df_train, df_test, cols):
    for col in cols:
        enc = get_encoder(df_train, col)
        df_train[col] = df_train[col].astype(str).apply(lambda x: enc.get(x, np.nan))
        df_test[col] = df_test[col].astype(str).apply(lambda x: enc.get(x, np.nan))
    return df_train, df_test


df_train, df_test = encode_all(df_train, df_test, encode_cols)
df_train, df_test = df_train, df_test
df_train.head()

clf = xgb.XGBClassifier(n_estimators=90, max_depth=4, random_state=42, colsample_bytree=0.5, colsample_bylevel=0.5, learning_rate=0.09, subsample=0.9)

X_train, y_train = df_train.drop("default", axis=1), df_train["default"]
X_test = df_test

# evaluation 101
skf = StratifiedKFold(y_train, 3, shuffle=True, random_state=42)

aucs = []
for (fold, (i_train, i_test)) in enumerate(skf):
    clf.fit(X_train.iloc[i_train], y_train.iloc[i_train])
    i_pred_proba = clf.predict_proba(X_train.iloc[i_test])
    print(i_pred_proba.shape)
    auc = roc_auc_score(y_train.iloc[i_test], i_pred_proba[:, 1])
    aucs.append(auc)
    print("AUC score on fold %i: %2.3f" % (fold, auc))
print("AUC: %2.3f +- %2.4f" % (np.mean(aucs), np.std(aucs)))
clf.fit(X_train, y_train)
sub = pd.DataFrame(clf.predict_proba(X_test)[:, 1], columns=["prob"], index=X_test.index)
sub.to_csv("submission101_xgb.csv")
sub
# Any results you write to the current directory are saved as output.