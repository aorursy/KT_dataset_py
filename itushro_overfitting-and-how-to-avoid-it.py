import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
X_train_original = train[["0", "1", "2", "3", "4"]]

y_train_original = train["target"]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train_original, y_train_original, test_size=0.2)

print(X_train.shape, X_test.shape)
from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=2)

clf.fit(X_train, y_train)
from sklearn.metrics import log_loss



probas = clf.predict_proba(X_train)

ll = log_loss(y_train, probas)



print("Log loss: {0}".format(ll))
probas = clf.predict_proba(X_test)

ll = log_loss(y_test, probas)



print("Log loss: {0}".format(ll))
for n in range(5):

    print("="*40)

    print("Neighbours: {0}".format(n+1))

    

    clf = KNeighborsClassifier(n_neighbors=n+1)

    clf.fit(X_train, y_train)



    probas = clf.predict_proba(X_train)

    ll = log_loss(y_train, probas)

    print("Train log loss: {0}".format(ll))

    

    probas = clf.predict_proba(X_test)

    ll = log_loss(y_test, probas)

    print("Test log loss: {0}".format(ll))
clf = KNeighborsClassifier(n_neighbors=5)

clf.fit(X_train_original, y_train_original)

test_probas = clf.predict_proba(test[["0", "1", "2", "3", "4"]])
sample_submission["target"] = test_probas[:,1]

sample_submission.to_csv("knn_baseline.csv",index=False)