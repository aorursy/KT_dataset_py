import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path
# Lets go ahead  and have a look at data

DATA_PATH = "../input/santander-customer-transaction-prediction/"  



train = pd.read_csv(str(Path(DATA_PATH) / "train.csv"))

test = pd.read_csv(str(Path(DATA_PATH) / "test.csv"))



print("Train and test shapes", train.shape, test.shape)
train.columns, test.columns
train.target.value_counts()
# https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

def woe(X, y):

    tmp = pd.DataFrame()

    tmp["variable"] = X

    tmp["target"] = y

    var_counts = tmp.groupby("variable")["target"].count()

    var_events = tmp.groupby("variable")["target"].sum()

    var_nonevents = var_counts - var_events

    tmp["var_counts"] = tmp.variable.map(var_counts)

    tmp["var_events"] = tmp.variable.map(var_events)

    tmp["var_nonevents"] = tmp.variable.map(var_nonevents)

    events = sum(tmp["target"] == 1)

    nonevents = sum(tmp["target"] == 0)

    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))

    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)

    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]

    iv = tmp.groupby("variable")["iv"].last().sum()

    return tmp["woe"], tmp["iv"], iv
iv_values = []

feats = ["var_{}".format(i) for i in range(200)]

y = train["target"]

for f in feats:

    X = pd.qcut(train[f], 10, duplicates='drop')

    _, _, iv = woe(X, y)

    iv_values.append(iv)

    

iv_inds = np.argsort(iv_values)[::-1][:50]

iv_values = np.array(iv_values)[iv_inds]

feats = np.array(feats)[iv_inds]

plt.figure(figsize=(10, 16))

sns.barplot(y=feats, x=iv_values, orient='h')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
feats = ["var_{}".format(i) for i in range(200)]

X = train[feats]

X_test = test[feats]

y = train["target"]



cvlist = list(StratifiedKFold(5, random_state=12345786).split(X, y))

scaler = StandardScaler()



X_sc = scaler.fit_transform(X)

X_test_sc = scaler.fit_transform(X_test)



lr = LogisticRegression()

y_preds_lr = cross_val_predict(lr, X_sc, y, cv=cvlist, method="predict_proba")[:, 1]



lr.fit(X_sc, y)

y_test_preds_lr = lr.predict_proba(X_test_sc)[:, 1] 

roc_auc_score(y, y_preds_lr)
sns.distplot(y_preds_lr)

sns.distplot(y_test_preds_lr)

plt.show()
import lightgbm as lgb

#model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.1, num_leaves=2, subsample=0.4, colsample_bytree=0.4)



#y_preds_lgb = np.zeros((len(y)))

#for i, (tr_idx, val_idx) in enumerate(cvlist):

#    X_dev, y_dev = X.iloc[tr_idx], y.iloc[tr_idx]

#    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

#    model.fit(X_dev, y_dev, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=50, early_stopping_rounds=200)

#    val_preds = model.predict_proba(X_val)[:, 1]

#    y_preds_lgb[val_idx] = val_preds

#    print("Score for fold {} is {}".format(i, roc_auc_score(y_val, val_preds)))

    

#print("Overall Score for oof predictions ", roc_auc_score(y, y_preds_lgb))
#model = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.1, num_leaves=8, subsample=0.6, colsample_bytree=0.6)

#model.fit(X, y)

#y_test_preds_lgb = model.predict_proba(X_test)[:, 1]

#sns.distplot(y_preds)

#sns.distplot(y_test_preds_lgb)
from scipy.stats import gmean
np.mean([0.9, 0.9, 0.9, 0.98, 0.9])
gmean([0.9, 0.9, 0.9, 0.98, 0.9])
import lightgbm as lgb

model = lgb.LGBMClassifier(n_estimators=200000, learning_rate=0.05, num_leaves=2, subsample=0.45, colsample_bytree=0.45)



y_preds_lgb = np.zeros((len(y)))

test_preds_allfolds = []

for i, (tr_idx, val_idx) in enumerate(cvlist):

    X_dev, y_dev = X.iloc[tr_idx], y.iloc[tr_idx]

    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model.fit(X_dev, y_dev, eval_set=[(X_val, y_val)], eval_metric="auc", verbose=50, early_stopping_rounds=200)

    val_preds = model.predict_proba(X_val)[:, 1]

    test_preds = model.predict_proba(X_test)[:, 1]

    test_preds_allfolds.append(test_preds)

    y_preds_lgb[val_idx] = val_preds

    print("Score for fold {} is {}".format(i, roc_auc_score(y_val, val_preds)))

    # break

print("Overall Score for oof predictions ", roc_auc_score(y, y_preds_lgb))
y_test_preds_lgb = gmean(test_preds_allfolds, 0)

sns.distplot(y_preds_lgb)

sns.distplot(y_test_preds_lgb)
sub = test[["ID_code"]]

sub["target"] = y_test_preds_lgb2

sub.to_csv("submission_lgbm2_v1.csv", index=False)
weighted_preds = y_preds_lr* 0.05 + y_preds * 0.95

roc_auc_score(y, weighted_preds)
public_sub = pd.read_csv("../input/santander-lgb-new-features-rank-mean-10-folds/submission_LGBM.csv")

public_sub.head()
sub["target"] = 0.1*sub["target"] + 0.9*public_sub["target"]

sub.to_csv("submission_blend.csv", index=False)