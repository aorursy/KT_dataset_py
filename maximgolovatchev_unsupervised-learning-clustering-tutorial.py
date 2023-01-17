import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from sklearn import datasets

from sklearn.cluster import KMeans, MeanShift, DBSCAN, Birch

from sklearn import metrics
data = datasets.load_digits()

X, y = data.data, data.target

X.shape, y.shape
sns.countplot(y);
plt.figure(figsize=(16, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X[i,:].reshape([8,8]), cmap='gray');
output = pd.DataFrame(index=['K-Means','Mean-Shift','DBSCAN','Birch'],

                      columns=['ARI','MI','HCV','FM','SC','CH','DB'])
# Fitting K-Means to data

clust_model = KMeans(n_clusters=10, random_state=17)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['K-Means','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['K-Means','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['K-Means','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['K-Means','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['K-Means','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['K-Means','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['K-Means','DB'] = metrics.davies_bouldin_score(X, labels)
result = []

for bw in tqdm_notebook(np.linspace(5,50,10)):

    clust_model = MeanShift(bandwidth=bw)

    clust_model.fit(X)

    labels = clust_model.labels_

    result.append(metrics.adjusted_rand_score(y, labels))

res = pd.DataFrame(index=np.linspace(5,50,10))

res['Score'] = result

res
result = []

for bw in tqdm_notebook(np.linspace(21,29,9)):

    clust_model = MeanShift(bandwidth=bw)

    clust_model.fit(X)

    labels = clust_model.labels_

    result.append(metrics.adjusted_rand_score(y, labels))

res = pd.DataFrame(index=np.linspace(21,29,9))

res['Score'] = result

res
# Fitting Mean-Shift to data

clust_model = MeanShift(bandwidth=26)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['Mean-Shift','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['Mean-Shift','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['Mean-Shift','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['Mean-Shift','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['Mean-Shift','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['Mean-Shift','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['Mean-Shift','DB'] = metrics.davies_bouldin_score(X, labels)
# Fitting DBSCAN to data

clust_model = DBSCAN(min_samples=2, eps=10)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['DBSCAN','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['DBSCAN','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['DBSCAN','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['DBSCAN','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['DBSCAN','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['DBSCAN','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['DBSCAN','DB'] = metrics.davies_bouldin_score(X, labels)
# Fitting Birch to data

clust_model = Birch(n_clusters=10)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['Birch','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['Birch','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['Birch','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['Birch','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['Birch','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['Birch','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['Birch','DB'] = metrics.davies_bouldin_score(X, labels)
output