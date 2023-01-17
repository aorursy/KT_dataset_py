import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.imputation import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
import os
print(os.listdir("../input/"))

to_drop = ["job_name", "reason"]

# have to upload train by hand since for some reason it is nont available here
df_train = pd.read_csv("../input/train_data.csv").set_index("ids").drop(to_drop, axis=1)
df_train = df_train[~df_train.default.isnull()]
df_train["default"] = df_train["default"].astype("int")
df_test = pd.read_csv("../input/teste_data.csv").set_index("ids").drop(to_drop, axis=1)
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
        df_train[col] = df_train[col].astype(str).apply(lambda x: enc.get(x, -1))
        df_test[col] = df_test[col].astype(str).apply(lambda x: enc.get(x, -1))
    return df_train, df_test
df_train, df_test = encode_all(df_train, df_test, encode_cols)
df_train, df_test = df_train.fillna(-1), df_test.fillna(-1)
df_train.head()
clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, max_features=10, oob_score=True)

X_train, y_train = df_train.drop("default", axis=1), df_train["default"]
X_test = df_test
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
sub.to_csv("submission101.csv")
sub
