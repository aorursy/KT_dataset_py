# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

%matplotlib inline

sns.set()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.head()
X = iris.iloc[:,1:5].values

y = iris.iloc[:,5:6].astype('category').values
from sklearn.decomposition import PCA

n_components=X.shape[1]

pca = PCA(n_components=n_components, random_state=123)

pca.fit(X)
explained_variance_ratio = pca.explained_variance_ratio_ 

cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

lst = []

for i in range (0, n_components):

    lst.append([i+1, round(explained_variance_ratio[i],6), cum_explained_variance_ratio[i]])



pca_predictor = pd.DataFrame(lst)

pca_predictor.columns = ['Component', 'Explained Variance', 'Cumulative Explained Variance']

pca_predictor
plt.subplots(figsize=(10,8))



plt.bar(pca_predictor['Component'],pca_predictor['Explained Variance'], align='center', alpha=0.5, 

       label='individual explained variance')

plt.step(pca_predictor['Component'],pca_predictor['Cumulative Explained Variance'], where='mid',

         label='cumulative explained variance')

plt.xlabel('Principal components')

plt.ylabel('Explained variance ratio');

plt.legend(loc='best')
pca = PCA(n_components=2)

pca.fit(X)



X_transformed = pca.fit_transform(X)



dfpc = pd.DataFrame(X_transformed, columns=['pc1', 'pc2'])

dfpc['class'] = y
dfpc
plt.subplots(figsize=(10,10))

sns.scatterplot(data=dfpc, x="pc1", y="pc2", hue='class', palette="Set2", s=100)
from sklearn.cluster import KMeans



n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=123)

kmeans.fit(X_transformed)
cluster_labels = kmeans.labels_

dfpc['KMeans_cluster'] = cluster_labels
plt.subplots(figsize=(10,10))

sns.scatterplot(data=dfpc, x="pc1", y="pc2", hue='KMeans_cluster', palette="Set2", s=100)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3,covariance_type='full')

gmm.fit(X_transformed)
labels = gmm.predict(X_transformed)

dfpc['GMM_cluster'] = labels
plt.subplots(figsize=(10,10))

sns.scatterplot(data=dfpc, x="pc1", y="pc2", hue='GMM_cluster', palette="Set2", s=100)
# find out which point has competed probabilites of clusters

probs = gmm.predict_proba(X_transformed)

probs = probs.round(3)

probs