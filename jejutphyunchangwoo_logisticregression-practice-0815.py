import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv("../input/avazu-ctr-prediction/train.gz", nrows=100000)

unused_cols = ["id", "hour", "device_id", "device_ip"]

label_col = "click"

train_df = train_df.drop(unused_cols, axis=1)

X_dict_train = list(train_df.drop(label_col, axis=1).T.to_dict().values())

y_train = train_df[label_col]
test_df = pd.read_csv("../input/avazu-ctr-prediction/train.gz", header=0, skiprows=(1,100000), nrows=100000)

test_df = test_df.drop(unused_cols, axis=1)

X_dict_test = list(test_df.T.to_dict().values())

y_test = test_df[label_col]
from sklearn.feature_extraction import DictVectorizer



vectorizer = DictVectorizer(sparse=True)

X_train = vectorizer.fit_transform(X_dict_train)

X_test = vectorizer.transform(X_dict_test)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

clf.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV



parameters = {'C': [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}

grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=3, scoring="roc_auc")

grid_search.fit(X_train, y_train)
grid_search.best_params_
clf_best = grid_search.best_estimator_
y_pred = clf_best.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve



accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)
y_pred_proba = clf_best.predict_proba(X_test)
y_pred_proba[:,0]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,0])

auc = roc_auc_score(y_test, y_pred_proba[:,0])

plt.plot(fpr, tpr, "r-", label="LogisticRegression")
auc