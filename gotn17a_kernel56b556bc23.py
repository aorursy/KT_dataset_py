!pip install lightgbm
import numpy as np

import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, accuracy_score, recall_score
import pandas as pd

X_test = pd.read_csv("../input/X_test.csv")

X_train = pd.read_csv("../input/X_train.csv")

submission = pd.read_csv("../input/submission.csv")

y_test_sample = pd.read_csv("../input/y_test_sample.csv")

y_train = pd.read_csv("../input/y_train.csv")
X_train = pd.read_csv("../input/X_train.csv")

y_train = pd.read_csv("../input/y_train.csv")

y_test_sample = pd.read_csv("../input/y_test_sample.csv", index_col="index")

X_test = pd.read_csv("../input/X_test.csv", index_col="index")
def preprocess(X_train, X_test, y_train):

    data = X_train.copy()

    data["target"] = y_train.target

    

    data.drop_duplicates(inplace=True)

    

    data.family_members.fillna(data.family_members.median(), inplace=True)

    X_test.family_members.fillna(X_test.family_members.median(), inplace=True)

    median = data.monthly_income.median()

    data.monthly_income.fillna(median, inplace=True)

    X_test.monthly_income.fillna(median, inplace=True)

    

    X_train = data.drop(columns=["target"])

    y_train = data.target

    return X_train, X_test, y_train
X_train, X_test, y_train = preprocess(X_train, X_test, y_train)
def metrics(clf, X, y, need_scaler=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

    if need_scaler:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)

    prob = clf.predict_proba(X_test)[:,1]

    print("F1 = {0}".format(f1_score(y_test, prediction)))

    print("Accuracy = {0}".format(accuracy_score(y_test, prediction)))

    print("Precision = {0}".format(precision_score(y_test, prediction)))

    print("Recall = {0}".format(recall_score(y_test, prediction)))

    print("ROC-AUC = {0}".format(roc_auc_score(y_test, prob)))
clf = LGBMClassifier(colsample_bytree=0.7, n_jobs=-1,

                   subsample=0.7,n_estimators=4500, learning_rate=0.0075, max_height=-1, metric='auc', verbose=10)
metrics(clf, X_train, y_train)
clf.fit(X_train, y_train)
def create_submission(clf, X_test, filename="submission.csv", scaler=None):

  if scaler is not None:

    X_test = scaler.transform(X_test)

  prediction = clf.predict_proba(X_test)[:, 1]

  prediction = pd.DataFrame({"index":y_test_sample.index, "target":prediction})

  prediction.set_index("index", inplace=True)

  prediction.to_csv(filename)
create_submission(clf, X_test)