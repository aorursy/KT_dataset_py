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
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(784,784,784,784,784))
mlp.fit(X_train,Y_train)
Y_pred = mlp.predict(X_test)
mlp.score(X_train, Y_train)
submission = pd.DataFrame({
        "ImageId": list(range(1, len(Y_pred)+1)),
        "Label": Y_pred
    })
submission.to_csv('submission.csv', index=False)
