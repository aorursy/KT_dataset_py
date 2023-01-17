import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA,KernelPCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
complete_features = pd.read_csv("../input/cached-features/complete_features.csv", parse_dates=["first_active_month"])
complete_features_test = pd.read_csv("../input/cached-features/complete_features_test.csv", parse_dates=["first_active_month"])
target = complete_features['target']
drops = ['card_id', 'first_active_month', 'target']
# to_remove = [c for c in complete_features if 'new' in c]
use_cols = [c for c in complete_features.columns if c not in drops]
features = list(complete_features[use_cols].columns)
complete_features[features] = complete_features[features].fillna(0)
complete_features_test[features] = complete_features_test[features].fillna(0)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(complete_features[features])
print(reduced_data.shape)
print(pca.explained_variance_ratio_)
plt.plot(reduced_data[:,0],reduced_data[:,1],'.b')
plt.show()
reduced_df = pd.DataFrame({'x' : reduced_data[:,0], 'y' : reduced_data[:,1], 'c': complete_features.target.apply(lambda x : x<-20).astype(int)})
plt.scatter(x=reduced_df['x'], y=reduced_df['y'],c=reduced_df['c'], s=1)
kpca = KernelPCA(n_components=2, kernel='rbf')
np.random.seed(3)
s = pd.concat([complete_features[complete_features['target']>-20].sample(2207),complete_features[complete_features['target']<-20]]).reset_index(drop='index')

reduced_data_d = kpca.fit_transform(s[features])
print(reduced_data_d.shape)
plt.plot(reduced_data_d[:,0],reduced_data_d[:,1],'.b')
reduced_df_d = pd.DataFrame({'x' : reduced_data_d[:,0], 'y' : reduced_data_d[:,1], 'c': s.target.apply(lambda x : x<-20).astype(int)})
plt.scatter(x=reduced_df_d['x'], y=reduced_df_d['y'],c=reduced_df_d['c'], s=1)
X_train, X_test, y_train, y_test = train_test_split(
    reduced_df_d[['x','y']].values, s.target.apply(lambda x : -1 if x<-20 else 1).values, test_size=0.2, random_state=42)
svm = SVC(C=1,gamma=1.3, tol=0.0001, kernel='rbf', degree=5)
svm.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100,max_depth=5)
rf.fit(X_train, y_train)

def make_meshgrid(x, y, h=.002):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    if not isinstance(clf, RandomForestClassifier):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out
    else:
        for tree in clf.estimators_:
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=1. /len(clf.estimators_), cmap=plt.cm.coolwarm)

    

fig, ax = plt.subplots(figsize=(12,10))
X0, X1 = X_train[:,0],X_train[:,1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, svm, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=25, edgecolors='k')
print(metrics.classification_report(y_train, svm.predict(X_train)))
fig, ax = plt.subplots(figsize=(12,10))

plot_contours(ax, rf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=1.0 / len(rf.estimators_))
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=25, edgecolors='k')
print(metrics.classification_report(y_train, rf.predict(X_train)))
