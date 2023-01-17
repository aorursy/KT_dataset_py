import pandas as pd



train_transaction = pd.read_csv("../input/train_transaction.csv")

train_identity = pd.read_csv("../input/train_identity.csv")

train_transaction.head()
train_identity.head()
train = train_transaction.join(train_identity, on="TransactionID", lsuffix="_leftid")

train.head()
temp = pd.DataFrame({

    "Columns": train.columns,

    "Types": train.dtypes

})

print(temp)
column_not_contains_missing = train.isna().sum() == 0

train.loc[:, train.columns[column_not_contains_missing]].head()
df = train.loc[:, train.columns[column_not_contains_missing]].drop(["TransactionID_leftid", "TransactionDT"], axis=1)



Y = df["isFraud"].copy()

X = pd.get_dummies(df.drop(["isFraud"], axis=1)).copy()

X.head()
print(X.shape)
import numpy as np



np.mean(Y)
import gc



del train_transaction, train_identity, train

gc.collect()
from sklearn.model_selection import cross_val_score



cv=9
from sklearn.ensemble import RandomForestClassifier





model = RandomForestClassifier(n_estimators=100)

scores_rf = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")

print(np.mean(scores_rf), "+/-", np.std(scores_rf))
from xgboost import XGBClassifier





model = XGBClassifier()

scores_xgb = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")

print(np.mean(scores_xgb), "+/-", np.std(scores_xgb))
from lightgbm import LGBMClassifier





model = LGBMClassifier()

scores_gbm = cross_val_score(model, X, Y, cv=cv, n_jobs=3, scoring="roc_auc")

print(np.mean(scores_gbm), "+/-", np.std(scores_gbm))
from catboost import CatBoostClassifier





X2 = (df.drop(["isFraud"], axis=1)).copy()



model = CatBoostClassifier(cat_features=["ProductCD"])

scores_cat = cross_val_score(model, X2, Y, cv=cv, n_jobs=3, scoring="roc_auc")

print(np.mean(scores_cat), "+/-", np.std(scores_cat))
classifier = ["Random forest"] * cv + ["Xgboost"] * cv + ["Lightgbm"] * cv + ["Catboost"] * cv

performance = pd.DataFrame({

    "classifier": classifier,

    "scores": list(scores_rf) + list(scores_xgb) + list(scores_gbm) + list(scores_cat)

})
import seaborn as sns

%matplotlib inline



sns.boxplot(y="classifier", x="scores", data=performance).set(xlabel='', ylabel='')