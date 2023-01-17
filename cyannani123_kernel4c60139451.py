import numpy as np 

import pandas as pd

import os

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler
train_df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

train_df.head()
test_df = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")

test_df.head()
total_df = train_df.append(test_df,sort=False).drop(columns=["target","id"])

X_cat = pd.get_dummies(total_df.drop(columns=["nom_5","nom_6","nom_7","nom_8","nom_9"])).fillna(0).astype("uint8")
X_cat.head()
X_train = X_cat[:600000]

X_test = X_cat[600000:]



y_train = train_df["target"]
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
clf = LogisticRegression(solver="saga",max_iter=200,random_state=0)

print("crossvalidation score:",cross_val_score(clf,X_train,y_train,cv=3,scoring='roc_auc').mean())
clf.fit(X_train,y_train)

submission = pd.DataFrame()

submission["id"] = test_df["id"]

submission["target"] = clf.predict_proba(X_test)[:,1]

pd.DataFrame(submission).to_csv("submission.csv",index=False)

print(pd.read_csv("submission.csv"))