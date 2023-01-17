from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
from sklearn import datasets
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC

import numpy as np
np.random.seed(42)
X, y = make_multilabel_classification(n_classes=2, n_labels=1, allow_unlabeled=True, random_state=1)

clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, y)
y_pred = clf.predict(X)
from sklearn.decomposition import PCA
trans = PCA(n_components=2)
X_trans = trans.fit_transform(X)

import matplotlib.pyplot as plt

where_0 = np.argwhere(y_pred[:, 0])
where_1 = np.argwhere(y_pred[:, 1])

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))
axarr[1].plot(X_trans[:, 0], X_trans[:, 1], marker='o', linewidth=0, markersize=4, color='black')
axarr[1].plot(X_trans[where_0, 0], X_trans[where_0, 1], marker='o', linewidth=0, markeredgewidth=3, markersize=8, markeredgecolor='steelblue', fillstyle='none')
axarr[1].plot(X_trans[where_1, 0], X_trans[where_1, 1], marker='o', linewidth=0, markeredgewidth=3, markersize=14, markeredgecolor='lightsteelblue', fillstyle='none')
axarr[1].set_title("$\{0, 1, None\}$-Labeled Points (Predicted)", fontsize=16)
pass

where_0 = np.argwhere(y[:, 0])
where_1 = np.argwhere(y[:, 1])

axarr[0].plot(X_trans[:, 0], X_trans[:, 1], marker='o', linewidth=0, markersize=4, color='black')
axarr[0].plot(X_trans[where_0, 0], X_trans[where_0, 1], marker='o', linewidth=0, markeredgewidth=3, markersize=8, markeredgecolor='steelblue', fillstyle='none')
axarr[0].plot(X_trans[where_1, 0], X_trans[where_1, 1], marker='o', linewidth=0, markeredgewidth=3, markersize=14, markeredgecolor='lightsteelblue', fillstyle='none')
axarr[0].set_title("$\{0, 1, None\}$-Labeled Points (Actual)", fontsize=16)

axarr[0].axis('off')
axarr[1].axis('off')
pass
from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
y_pred = MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)
from sklearn.metrics import r2_score
r2_score(y, y_pred)