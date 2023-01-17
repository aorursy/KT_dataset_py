# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib as mpl

import itertools

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

iris = pd.read_csv('/kaggle/input/iris-dataset/iris.data.csv', names=iris_columns)

iris.head()
# Extract the continuous attributes to a matrix X

X = iris.iloc[:,0:4].values
# Extract the class attribute as y

y = iris.iloc[:,4].astype('category').values
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)

pca.fit(X)

X_fit = pca.transform(X)
print('Cumulative Variance explained by 1 principal components: %.2f%%' % ( sum(pca.explained_variance_ratio_) * 100))
# Put principal components into a data frame so we can plot it.

dfpc = pd.DataFrame(X_fit, columns=['pc1', 'pc2'])

dfpc['class'] = y
plt.figure(1, figsize=(10,10), dpi=100)

plt.clf()

sns.lmplot(data=dfpc, x="pc1", y="pc2", fit_reg=False, hue='class')

plt.show()
X_fit.shape
X_fit
from sklearn.cluster import KMeans



n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, random_state=123)

kmeans.fit(X_fit)
cluster_labels = kmeans.labels_

cluster_labels
ax=plt.subplots(figsize=(10,5))

ax=sns.countplot(cluster_labels)

title="Histogram of Cluster Counts of K-means"

ax.set_title(title, fontsize=12)

plt.show()
iris['X'] = X_fit[:,[0]]

iris['Y'] = X_fit[:,[1]]

iris['cluster'] = cluster_labels
ax=plt.subplots(figsize=(10,10))

ax = sns.scatterplot(x='X', y='Y', hue='cluster', legend="full", palette="Set1", data=iris)
# Find the best Gaussian Mixture Model

from sklearn.mixture import GaussianMixture

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']



for cv_type in cv_types:

    for n_components in n_components_range:

        # Fit a Gaussian mixture

        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)

        gmm.fit(X_fit)

        bic.append(gmm.bic(X_fit))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm



bic = np.array(bic) # convert to Numpy array

print('Best model: {}'.format(best_gmm.covariance_type))
best_gmm.covariances_
# Plot unclustered and clustered points side by side

from scipy import linalg

color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

plt.figure(figsize=(12, 5))



ax1 = plt.subplot(1, 2, 1)

ax1.scatter(X_fit[:, 0], X_fit[:, 1], marker='.', c='k', s=20, edgecolor='')

ax1.set_title('Original Points')

ax1.set_xlim(-4, 4)

ax1.set_ylim(-2, 2)





# Plot the winner

splot = plt.subplot(1, 2, 2)

splot.set_xlim(-4, 4)

splot.set_ylim(-2, 2)

Y_fit = best_gmm.predict(X_fit)



for i, (mean, cov, color) in enumerate(zip(best_gmm.means_, best_gmm.covariances_, color_iter)):

    # Convert covariance to square matrix

    if best_gmm.covariance_type == 'full':

        cov = best_gmm.covariances_[i][:2, :2]

    elif best_gmm.covariance_type == 'tied':

        cov = best_gmm.covariances_[:2, :2]

    elif best_gmm.covariance_type == 'diag':

        cov = np.diag(best_gmm.covariances_[i][:2])

    elif best_gmm.covariance_type == 'spherical':

        cov = np.eye(best_gmm.means_.shape[1]) * best_gmm.covariances_[i]



    v, w = linalg.eigh(cov)

    if not np.any(Y_fit == i):

        continue

        

    plt.scatter(X_fit[Y_fit ==i, 0], X_fit[Y_fit ==i, 1], marker='.', s=20, edgecolor='', color=color)



    # Plot an ellipse to show the Gaussian component

    angle = np.arctan2(w[0][1], w[0][0])

    angle = 180. * angle / np.pi  # convert to degrees

    v = 2. * np.sqrt(2.) * np.sqrt(v)

    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)

    ell.set_clip_box(splot.bbox)

    ell.set_alpha(.5)

    splot.add_artist(ell)



plt.title('Selected GMM: {} model, {} components'.format(best_gmm.covariance_type, best_gmm.n_components))

plt.subplots_adjust(hspace=.35, bottom=.02)

plt.show()