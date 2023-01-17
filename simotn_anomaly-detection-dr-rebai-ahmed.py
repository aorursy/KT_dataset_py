%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import numpy as np

import matplotlib

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



X, y = make_blobs(n_features=2, centers=3, n_samples=500,

                  random_state=42)
X.shape
plt.figure()

plt.scatter(X[:, 0], X[:, 1])

plt.show()
from sklearn.neighbors.kde import KernelDensity



# Estimate density with a Gaussian kernel density estimator

kde = KernelDensity(kernel='gaussian')

kde = kde.fit(X)

kde
kde_X = kde.score_samples(X)

print(kde_X.shape)  # contains the log-likelihood of the data. The smaller it is the rarer is the sample
from scipy.stats.mstats import mquantiles

alpha_set = 0.95

tau_kde = mquantiles(kde_X, 1. - alpha_set)
n_samples, n_features = X.shape

X_range = np.zeros((n_features, 2))

X_range[:, 0] = np.min(X, axis=0) - 1.

X_range[:, 1] = np.max(X, axis=0) + 1.



h = 0.1  # step size of the mesh

x_min, x_max = X_range[0]

y_min, y_max = X_range[1]

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



grid = np.c_[xx.ravel(), yy.ravel()]
Z_kde = kde.score_samples(grid)

Z_kde = Z_kde.reshape(xx.shape)



plt.figure()

c_0 = plt.contour(xx, yy, Z_kde, levels=tau_kde, colors='red', linewidths=3)

plt.clabel(c_0, inline=1, fontsize=15, fmt={tau_kde[0]: str(alpha_set)})

plt.scatter(X[:, 0], X[:, 1])

plt.show()
from sklearn.svm import OneClassSVM
nu = 0.05  # theory says it should be an upper bound of the fraction of outliers

ocsvm = OneClassSVM(kernel='rbf', gamma=0.05, nu=nu)

ocsvm.fit(X)
X_outliers = X[ocsvm.predict(X) == -1]
Z_ocsvm = ocsvm.decision_function(grid)

Z_ocsvm = Z_ocsvm.reshape(xx.shape)



plt.figure()

c_0 = plt.contour(xx, yy, Z_ocsvm, levels=[0], colors='red', linewidths=3)

plt.clabel(c_0, inline=1, fontsize=15, fmt={0: str(alpha_set)})

plt.scatter(X[:, 0], X[:, 1])

plt.scatter(X_outliers[:, 0], X_outliers[:, 1], color='red')

plt.show()
X_SV = X[ocsvm.support_]

n_SV = len(X_SV)

n_outliers = len(X_outliers)



print('{0:.2f} <= {1:.2f} <= {2:.2f}?'.format(1./n_samples*n_outliers, nu, 1./n_samples*n_SV))
plt.figure()

plt.contourf(xx, yy, Z_ocsvm, 10, cmap=plt.cm.Blues_r)

plt.scatter(X[:, 0], X[:, 1], s=1.)

plt.scatter(X_SV[:, 0], X_SV[:, 1], color='orange')

plt.show()
# %load solutions/22_A-anomaly_ocsvm_gamma.py



nu = 0.05  # theory says it should be an upper bound of the fraction of outliers



for gamma in [0.001, 1.]:

    ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)

    ocsvm.fit(X)



    Z_ocsvm = ocsvm.decision_function(grid)

    Z_ocsvm = Z_ocsvm.reshape(xx.shape)



    plt.figure()

    c_0 = plt.contour(xx, yy, Z_ocsvm, levels=[0], colors='red', linewidths=3)

    plt.clabel(c_0, inline=1, fontsize=15, fmt={0: str(alpha_set)})

    plt.scatter(X[:, 0], X[:, 1])

    plt.scatter(X_outliers[:, 0], X_outliers[:, 1], color='red')

    plt.legend()

    plt.show()

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=300, contamination=0.10)

iforest = iforest.fit(X)
### Z_iforest = iforest.decision_function(grid)

Z_iforest = Z_iforest.reshape(xx.shape)



plt.figure()

c_0 = plt.contour(xx, yy, Z_iforest,

                  levels=[iforest.threshold_],

                  colors='red', linewidths=3)

plt.clabel(c_0, inline=1, fontsize=15,

           fmt={iforest.threshold_: str(alpha_set)})

plt.scatter(X[:, 0], X[:, 1], s=1.)

plt.show()
# %load solutions/22_B-anomaly_iforest_n_trees.py

for n_estimators in [1, 10, 50, 100]:

    iforest = IsolationForest(n_estimators=n_estimators, contamination=0.10)

    iforest = iforest.fit(X)



    Z_iforest = iforest.decision_function(grid)

    Z_iforest = Z_iforest.reshape(xx.shape)



    plt.figure()

    c_0 = plt.contour(xx, yy, Z_iforest,

                      levels=[iforest.threshold_],

                      colors='red', linewidths=3)

    plt.clabel(c_0, inline=1, fontsize=15,

               fmt={iforest.threshold_: str(alpha_set)})

    plt.scatter(X[:, 0], X[:, 1], s=1.)

    plt.legend()

    plt.show()
from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images

labels = digits.target

images.shape
i = 102



plt.figure(figsize=(2, 2))

plt.title('{0}'.format(labels[i]))

plt.axis('off')

plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
n_samples = len(digits.images)

data = digits.images.reshape((n_samples, -1))
data.shape
X = data

y = digits.target
X.shape
X_5 = X[y == 5]
X_5.shape
fig, axes = plt.subplots(1, 5, figsize=(10, 4))

for ax, x in zip(axes, X_5[:5]):

    img = x.reshape(8, 8)

    ax.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')

    ax.axis('off')
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(contamination=0.05)

iforest = iforest.fit(X_5)

iforest_X = iforest.decision_function(X_5)

plt.hist(iforest_X);
X_strong_inliers = X_5[np.argsort(iforest_X)[-10:]]



fig, axes = plt.subplots(2, 5, figsize=(10, 5))



for i, ax in zip(range(len(X_strong_inliers)), axes.ravel()):

    ax.imshow(X_strong_inliers[i].reshape((8, 8)),

               cmap=plt.cm.gray_r, interpolation='nearest')

    ax.axis('off')
fig, axes = plt.subplots(2, 5, figsize=(10, 5))



X_outliers = X_5[iforest.predict(X_5) == -1]



for i, ax in zip(range(len(X_outliers)), axes.ravel()):

    ax.imshow(X_outliers[i].reshape((8, 8)),

               cmap=plt.cm.gray_r, interpolation='nearest')

    ax.axis('off')
# %load solutions/22_C-anomaly_digits.py

k = 1  # change to see other numbers



X_k = X[y == k]



iforest = IsolationForest(contamination=0.05)

iforest = iforest.fit(X_k)

iforest_X = iforest.decision_function(X_k)



X_strong_outliers = X_k[np.argsort(iforest_X)[:10]]



fig, axes = plt.subplots(2, 5, figsize=(10, 5))



for i, ax in zip(range(len(X_strong_outliers)), axes.ravel()):

    ax.imshow(X_strong_outliers[i].reshape((8, 8)),

               cmap=plt.cm.gray_r, interpolation='nearest')

    ax.axis('off')
