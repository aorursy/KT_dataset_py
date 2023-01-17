import numpy as np
import pandas as pd

import sys
import os
print(os.listdir("../input"))

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
X_train = train.drop("label",axis=1)
Y_train = train["label"]
X_test  = test.copy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
clf_gini = DecisionTreeClassifier(criterion = "entropy",random_state = 100,max_depth=10, min_samples_leaf=5)
clf_gini.fit(X_train, Y_train)
Y_pred = clf_gini.predict(X_test)
clf_gini.score(X_train, Y_train)
submission = pd.DataFrame({
        "ImageId": list(range(1, len(Y_pred)+1)),
        "Label": Y_pred
    })
submission.to_csv('submission.csv', index=False)