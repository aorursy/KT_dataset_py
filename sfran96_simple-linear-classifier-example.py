import numpy as np

import pandas as pd

from scipy.io import loadmat

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from seaborn import heatmap

from sklearn.decomposition import PCA
data = pd.read_csv("../input/raman-spectroscopy-for-detecting-covid19/covid_and_healthy_spectra.csv")

data.head()
X, X_test, y, y_test = train_test_split(data.loc[:, "400":"2112"], data.loc[:, "diagnostic"], shuffle=True, random_state=7, test_size=0.2)

svc = LinearSVC(penalty="l2", loss="squared_hinge")

svc.fit(X, y)

heatmap(confusion_matrix(svc.predict(X_test), y_test), annot=True, fmt="g", yticklabels=["Healthy", "SARS-CoV-19"] ,xticklabels=["Healthy", "SARS-CoV-19"])

plt.title("Confusion matrix for test-dataset")

plt.xlabel("Predicted")

plt.ylabel("Expected")

plt.figure()

heatmap(confusion_matrix(svc.predict(X), y), annot=True, fmt="g", yticklabels=["Healthy", "SARS-CoV-19"] ,xticklabels=["Healthy", "SARS-CoV-19"])

plt.title("Confusion matrix for train-dataset")

plt.xlabel("Predicted")

plt.ylabel("Expected")

print(f"Classification report for test-dataset:\n{classification_report(svc.predict(X_test), y_test)}")

print(f"Classification report for train-dataset:\n{classification_report(svc.predict(X), y)}")
pca = PCA()

pca.fit(X)

print("Explained variance > 0.95:", (pca.explained_variance_ratio_.cumsum() > 0.95)[0:5], "...")

plt.figure(figsize=(14,8))

plt.plot(data.columns[:-1].astype(int), np.abs(pca.components_[0:5]).sum(axis=0))

plt.show()
plt.figure(figsize=(16,8))

plt.plot(data.columns[:-1].astype(int), data.iloc[:, :-1].std())

plt.title("Weights of the first five components for PCA on the std. dev. of the whole dataset")

plt.xlabel("Raman shift ($cm^{-1}$)")

plt.ylabel("Intensity (Arbitrary units)")

weights = np.abs(pca.components_[0:5]).sum(axis=0) / np.abs(pca.components_[0:5]).sum(axis=0).max()

ax = plt.gca()

cb = ax.pcolorfast((data.columns[:-1].astype(int).min(), data.columns[:-1].astype(int).max()), (data.iloc[0, :-1].min(),data.iloc[0, :-1].max()), (weights, weights), alpha=0.5, cmap="binary")

plt.colorbar(cb)

plt.show()