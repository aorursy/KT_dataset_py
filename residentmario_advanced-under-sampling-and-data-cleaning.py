import seaborn as sns
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.01, 0.05, 0.94],
                           class_sep=0.8, random_state=0)

import matplotlib.pyplot as plt
colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
plt.scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)
sns.despine()
from imblearn.under_sampling import ClusterCentroids
trans = ClusterCentroids(random_state=0)
X_resampled, y_resampled = trans.fit_sample(X, y)

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
sns.despine()
from imblearn.under_sampling import NearMiss
trans = NearMiss(version=1)
X_resampled, y_resampled = trans.fit_sample(X, y)

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
sns.despine()
from imblearn.under_sampling import NearMiss
trans = NearMiss(version=2)
X_resampled, y_resampled = trans.fit_sample(X, y)

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
sns.despine()
from imblearn.under_sampling import NearMiss
trans = NearMiss(version=3)
X_resampled, y_resampled = trans.fit_sample(X, y)

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
sns.despine()
import numpy as np
sampler = TomekLinks(random_state=0)

# minority class
X_minority = np.transpose([[1.1, 1.3, 1.15, 0.8, 0.55, 2.1],
                           [1., 1.5, 1.7, 2.5, 0.55, 1.9]])
# majority class
X_majority = np.transpose([[2.1, 2.12, 2.13, 2.14, 2.2, 2.3, 2.5, 2.45],
                           [1.5, 2.1, 2.7, 0.9, 1.0, 1.4, 2.4, 2.9]])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax_arr = (ax1, ax2)
title_arr = ('Removing only majority samples',
             'Removing all samples')
for ax, title, sampler in zip(ax_arr,
                              title_arr,
                              [TomekLinks(ratio='auto', random_state=0),
                               TomekLinks(ratio='all', random_state=0)]):
    X_res, y_res = sampler.fit_sample(np.vstack((X_minority, X_majority)),
                                      np.array([0] * X_minority.shape[0] +
                                               [1] * X_majority.shape[0]))
    ax.scatter(X_res[y_res == 0][:, 0], X_res[y_res == 0][:, 1],
               label='Minority class', s=200, marker='_')
    ax.scatter(X_res[y_res == 1][:, 0], X_res[y_res == 1][:, 1],
               label='Majority class', s=200, marker='+')

    # highlight the samples of interest
    ax.scatter([X_minority[-1, 0], X_majority[1, 0]],
               [X_minority[-1, 1], X_majority[1, 1]],
               label='Tomek link', s=200, alpha=0.3)

    ax.set_title(title)
#     make_plot_despine(ax)
fig.tight_layout()

plt.show()
from imblearn.under_sampling import TomekLinks
trans = TomekLinks(ratio='all')
X_resampled, y_resampled = trans.fit_sample(X, y)

plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
sns.despine()
len(X), len(X_resampled)
_X_minority = X[np.where(y == 0)[0]]
plt.scatter(_X_minority[:, 0], _X_minority[:, 1], c='white', linewidth=1, edgecolor='red')

trans = TomekLinks(ratio='all')
X_resampled, y_resampled = trans.fit_sample(X, y)
_X_minority = X_resampled[np.where(y_resampled == 0)[0]]
plt.scatter(_X_minority[:, 0], _X_minority[:, 1], c='lightgreen', edgecolor='green', linewidth=1)
plt.suptitle("Tomek Links Removed from Minority Class")
pass
from imblearn.under_sampling import EditedNearestNeighbours

fig, axarr = plt.subplots(2, 2, figsize=(12, 12))
kwarg_params = {'edgecolor': 'darkgray'}

trans = EditedNearestNeighbours()
X_resampled, y_resampled = trans.fit_sample(X, y)
X_diff = 5000 - len(X_resampled)
print(X_diff)
plt.sca(axarr[0][0])
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
axarr[0][0].set_title("$kind=all, k=3$, $\Delta n =-155$")

trans = EditedNearestNeighbours(n_neighbors=10)
X_resampled, y_resampled = trans.fit_sample(X, y)
X_diff = 5000 - len(X_resampled)
print(X_diff)
plt.sca(axarr[0][1])
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
axarr[0][1].set_title("$kind=all, k=10$, $\Delta n =-411$")

trans = EditedNearestNeighbours(n_neighbors=1, kind_sel='mode')
X_resampled, y_resampled = trans.fit_sample(X, y)
X_diff = 5000 - len(X_resampled)
print(X_diff)
plt.sca(axarr[1][0])
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
axarr[1][0].set_title("$kind=all, k=1$, $\Delta n =-68$")

trans = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode')
X_resampled, y_resampled = trans.fit_sample(X, y)
X_diff = 5000 - len(X_resampled)
print(X_diff)
plt.sca(axarr[1][1])
plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled], **kwarg_params)
axarr[1][1].set_title("$kind=all, k=10$, $\Delta n =-36$")
plt.suptitle("EditedNearestNeighbours Output w/ Various Settings")
pass