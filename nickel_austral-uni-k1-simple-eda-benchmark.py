import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score
%matplotlib inline 
data_path="../input"
os.listdir(data_path)
apps = pd.read_csv(os.path.join(data_path, "application_train.csv"), index_col="SK_ID_CURR")
apps.head()
display("## Simple Data Statistics")
pd.options.display.max_rows = 125
pd.concat([apps.dtypes.rename("dtypes"), apps.isnull().sum().rename("Nulls"), apps.apply(lambda x: x.unique().shape[0]).rename("Cardinality"), apps.describe().T], axis=1)
pd.options.display.max_rows = 50
from IPython.display import display

for c in apps.drop("TARGET", axis=1):
    card = apps[c].unique().shape[0]
    serie = apps[c].fillna(-99999999999999999)
    print("### " + c + " ###")
    if card <= 10:
        count = serie.value_counts().rename("cantidad")
        pos = pd.crosstab(serie, apps.TARGET).iloc[:, -1].rename("positivos")
    else: 
        for k in range(10, 0, -1):
            try:
                count = pd.qcut(serie, k).value_counts().rename("cantidad")
                pos = pd.crosstab(pd.qcut(serie, k), apps.TARGET).iloc[:, -1].rename("positivos")
                break
            except:
                continue
    res = pd.concat([count, pos], axis=1)
    res["densidad"] = res.positivos / res.cantidad
    display(res)
for c in apps.select_dtypes(include="object"):
    apps[c] = apps[c].astype("category")
apps.dtypes.value_counts()
from lightgbm import LGBMClassifier
def train_model(train, target, nl, X_test):
    test_probs = []
    for i in range(5):
        valid = train.sample(frac=0.1)
        X_valid = valid.drop(target, axis=1)
        y_valid = valid[target]
        X_train = train.drop(valid.index)
        y_train = X_train[target]
        X_train = X_train.drop(target, axis=1)

        learner = LGBMClassifier(n_estimators=10000, num_leaves=nl)
        learner.fit(X_train, y_train,  early_stopping_rounds=10, eval_metric="auc", verbose=25,
                    eval_set=[(X_train, y_train),
                              (X_valid, y_valid)])
        probs = pd.Series(learner.predict_proba(X_test)[:, -1], index=X_test.index, name="fold_" + str(i))
        test_probs.append(probs)
    return pd.concat(test_probs, axis=1).mean(axis=1)
test = apps.sample(frac=0.1)
train = apps.drop(test.index)
X_test = test.drop("TARGET", axis=1)
y_test = test.TARGET

nls = [2 ** i for i in [3, 4, 5, 6, 8]]
res = pd.Series([np.nan] * len(nls), index=nls, name="ROC_AUC")
for nl in nls:
    print("*"*10, nl, "*"*10)
    probs = train_model(train, "TARGET", nl, X_test)
    res.loc[nl] = roc_auc_score(y_test, probs)
    print("ROC_AUC for {nl} leaves: {res:.4f}".format(nl=nl, res=res.loc[nl]))
train = apps
X_test = pd.read_csv(os.path.join(data_path, "application_test.csv"), index_col="SK_ID_CURR")
for c in X_test.select_dtypes(include="object"):
    X_test[c] = X_test[c].astype("category")
train_model(train, "TARGET", res.idxmax(), X_test).rename("TARGET").to_csv("submission_bestNL.csv", header=True)