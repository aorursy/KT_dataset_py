import numpy as np

import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import matthews_corrcoef, roc_auc_score

from sklearn.cross_validation import cross_val_score, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
date_chunks = pd.read_csv("../input/train_date.csv", index_col=0, chunksize=1000, dtype=np.float32)

num_chunks = pd.read_csv("../input/train_numeric.csv", index_col=0,

                         usecols=list(range(969)), chunksize=1000, dtype=np.float32)

X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)

               for dchunk, nchunk in zip(date_chunks, num_chunks)])

y = pd.read_csv("../input/train_numeric.csv", index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()

X = X.values
clf = XGBClassifier(base_score=0.005)

clf.fit(X, y)
plt.hist(clf.feature_importances_[clf.feature_importances_>0])

important_indices = np.where(clf.feature_importances_>0.01)[0]

print(important_indices)
n_date_features = 1156

X = np.concatenate([

    pd.read_csv("../input/train_date.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices < n_date_features] + 1])).values,

    pd.read_csv("../input/train_numeric.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices >= n_date_features] + 1 - 1156])).values

], axis=1)

y = pd.read_csv("../input/train_numeric.csv", index_col=0, dtype=np.float32, usecols=[0,969]).values.ravel()
clf = XGBClassifier(max_depth=5, base_score=0.005)

cv = StratifiedKFold(y, n_folds=3)

preds = np.ones(y.shape[0])

for i, (train, test) in enumerate(cv):

    preds[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]

    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(y[test], preds[test])))

print(roc_auc_score(y, preds))
thresholds = np.linspace(0.01, 0.99, 50)

mcc = np.array([matthews_corrcoef(y, preds>thr) for thr in thresholds])

plt.plot(thresholds, mcc)

best_threshold = thresholds[mcc.argmax()]

print(mcc.max())
X = np.concatenate([

    pd.read_csv("../input/test_date.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices<1156]+1])).values,

    pd.read_csv("../input/test_numeric.csv", index_col=0, dtype=np.float32,

                usecols=np.concatenate([[0], important_indices[important_indices>=1156] +1 - 1156])).values

], axis=1)
preds = (clf.predict_proba(X)[:,1] > best_threshold).astype(np.int8)
sub = pd.read_csv("../input/sample_submission.csv", index_col=0)

sub["Response"] = preds

sub.to_csv("submission.csv.gz", compression="gzip")