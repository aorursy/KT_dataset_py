import numpy as np

import pandas as pd

import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing.imputation import Imputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
import os

print(os.listdir("../input/"))



to_drop = ["job_name", "reason", "zip"]

# have to upload train by hand since for some reason it is nont available here

df_train = pd.read_csv("../input/train_data_.csv").set_index("ids").drop(to_drop, axis=1)


df_train = df_train[~df_train.default.isnull()]

df_train["default"] = df_train["default"].astype("int")

df_prod = pd.read_csv("../input/data_no_label.csv").set_index("ids").drop(to_drop, axis=1)

print((df_train.shape, df_prod.shape))
# stats

print(pd.concat([df_train.isnull().mean(), df_train.dtypes, df_train.T.apply(lambda x: x.nunique(), axis=1)], axis=1))
encode_cols = df_train.dtypes

encode_cols = encode_cols[encode_cols == object].index.tolist()
encode_cols
for i in encode_cols:

    freq = dict(df_train[i].value_counts())

    df_train[i] = df_train[i].replace(freq).astype(float)

    df_prod[i] = df_prod[i].replace(freq).astype(float, errors = 'ignore')
df_train = df_train.fillna(-999)

df_prod = df_prod.fillna(-999)
from sklearn.model_selection import train_test_split

X, y = df_train.drop("default", axis=1), df_train["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_prod = df_prod

print(X_train.shape)

print(X_test.shape)

print(X_prod.shape)
tunning = {'max_depth':[5,10], 'max_features':[11,22]}



cv = GridSearchCV(RandomForestClassifier(random_state=42, n_estimators=200), tunning, cv=5, verbose=10, scoring='roc_auc',

                 n_jobs=-1)
cv.fit(X_train, y_train)
cv.best_params_
from sklearn.metrics import roc_auc_score

print(roc_auc_score(y_test, cv.predict_proba(X_test)[:, 1]))
sub = pd.DataFrame(cv.predict_proba(X_prod)[:, 1], columns=["prob"], index=X_prod.index)

sub.to_csv("submission_exemplo.csv")

sub