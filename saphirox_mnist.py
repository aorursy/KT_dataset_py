import numpy as np

import pandas as pd

import os

from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

import seaborn as sns

%pylab inline
digits = load_digits()
data = digits.data

data.shape
pylab.imshow(digits.images[1], cmap = 'gray', interpolation = 'nearest')


for plot_number, plot in enumerate(digits.images[:10]):

    pyplot.subplot(2, 5, plot_number + 1)

    pylab.imshow(plot, cmap = 'gray')

    pylab.title('digit: ' + str(digits.target[plot_number]))
labels = digits.target
from collections import Counter

plt.figure(figsize=(16, 8))



sns.barplot(x = list(Counter(labels).keys()), y = list(Counter(labels).values()))

np.var(list(Counter(labels).values()))
from sklearn.random_projection import SparseRandomProjection

from sklearn.model_selection import train_test_split



proj = SparseRandomProjection(n_components=2)

trans_proj = proj.fit_transform(data, labels)

plt.figure(figsize=(20, 5))

plt.scatter(trans_proj[:,0], trans_proj[:,1], c = labels, cmap='RdYlBu')
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)



rforest = RandomForestClassifier()

rforest.fit(X_train, y_train)

print(classification_report(rforest.predict(X_test), y_test))
X_train, X_test, y_train, y_test = train_test_split(trans_proj, labels, test_size=0.3)



rforest.fit(X_train, y_train)

print(classification_report(rforest.predict(X_test), y_test))
from sklearn.decomposition import PCA

pca = PCA(n_components=2, iterated_power=5)

trans_data = pca.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(trans_data, labels, test_size=0.3)

rforest.fit(X_train, y_train)

print(classification_report(rforest.predict(X_test), y_test))
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2)

trans_data = tsne.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(trans_data, labels, test_size=0.3)

rforest.fit(X_train, y_train)

print(classification_report(rforest.predict(X_test), y_test))
fig, ax = plt.subplots(figsize=(20,10))

ax.legend()

ax.scatter(trans_data[:, 0], trans_data[:, 1], c=labels, alpha=0.7)



ax.grid()