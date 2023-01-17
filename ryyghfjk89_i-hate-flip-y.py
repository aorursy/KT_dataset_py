import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from tqdm import tqdm_notebook

import warnings

import multiprocessing

from scipy.optimize import minimize  

import time

from sklearn.model_selection import GridSearchCV, train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification 

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

np.random.seed(2019)
X, y = make_classification(n_samples = 10000, flip_y=0.08, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)



plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=100, edgecolor="k", linewidth=2)



plt.xlabel("$X_1$")

plt.ylabel("$X_2$")
oof = np.zeros(len(y))

skf = StratifiedKFold(n_splits=11, random_state=42)

for train_index, val_index in skf.split(X, y):



    clf = QuadraticDiscriminantAnalysis()

    clf.fit(X,y)

    oof[val_index] = clf.predict_proba(X[val_index,:])[:,1]

    

print(roc_auc_score(y, oof))

print("")

print(confusion_matrix(y, clf.predict(X)))
X, y = make_classification(n_samples = 10000, flip_y=0.00, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)



plt.scatter(X[:, 1], X[:, 0], marker='o', c=y,

            s=100, edgecolor="k", linewidth=2)



plt.xlabel("$X_1$")

plt.ylabel("$X_2$")
oof = np.zeros(len(y))

skf = StratifiedKFold(n_splits=11, random_state=42)

for train_index, val_index in skf.split(X, y):



    clf = QuadraticDiscriminantAnalysis()

    clf.fit(X,y)

    oof[val_index] = clf.predict_proba(X[val_index,:])[:,1]

    

print(roc_auc_score(y, oof))

print("")

print(confusion_matrix(y, clf.predict(X)))
X, y = make_classification(n_samples = 10000, flip_y=1, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)



plt.scatter(X[:, 1], X[:, 0], marker='o', c=y,

            s=100, edgecolor="k", linewidth=2)



plt.xlabel("$X_1$")

plt.ylabel("$X_2$")
oof = np.zeros(len(y))

skf = StratifiedKFold(n_splits=11, random_state=42)

for train_index, val_index in skf.split(X, y):



    clf = QuadraticDiscriminantAnalysis()

    clf.fit(X,y)

    oof[val_index] = clf.predict_proba(X[val_index,:])[:,1]

    

print(roc_auc_score(y, oof))

print("")

print(confusion_matrix(y, clf.predict(X)))