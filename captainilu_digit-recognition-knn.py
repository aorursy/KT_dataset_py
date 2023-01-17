import numpy as np
import pandas as pd

import os
print(os.listdir("../input"))

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X_train = train.drop("label",axis=1)
Y_train = train["label"]
X_test  = test.copy()
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
submission = pd.DataFrame({
        "ImageId": list(range(1, len(Y_pred)+1)),
        "Label": Y_pred
    })
submission.to_csv('submission.csv', index=False)