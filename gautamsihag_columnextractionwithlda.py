%matplotlib inline
import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime
import time
from sklearn.decomposition import PCA
desti1 = pd.read_csv("../input/destinations.csv", nrows=10000)
X = desti1.ix[:,1:150]
y = desti1.ix[:,0:1]
y.info()
#target_names = iris.target_names
pca = PCA(n_components=5)
X_r = pca.fit(X).transform(X)
X_r1 = pd.DataFrame(X_r)
X_r1["srch_destination_id"] = desti1["srch_destination_id"]
# Percentage of variance explained for each components
print('explained variance ratio (first 5 components): %s' % str(pca.explained_variance_ratio_))
X_r1.head()
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
n_components = np.arange(0, 149, 5)
def compute_scores(X):
    pca = PCA( )
    fa = FactorAnalysis( )

    pca_scores, fa_scores = [ ], [ ]
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores
def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))
def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))
for X in X:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % 10, linestyle='-')
    plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.axvline(n_components_pca_mle, color='k', label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet', label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange', label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)

plt.show()