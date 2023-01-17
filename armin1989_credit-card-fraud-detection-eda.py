# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.manifold import SpectralEmbedding

from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")

data.describe()
data.head(20)
data.info()
plt.hist(data["Class"], bins=2)
data["Class"].value_counts()
#randomly selecting 442 random non-fraudulent transactin, but normalizing data before undersampling

fraud = data[data['Class'] == 1]

non_fraud = data[data['Class'] == 0].sample(len(fraud) * 5)

non_fraud.reset_index(drop=True, inplace=True)

fraud.reset_index(drop=True, inplace=True)

new_data = pd.concat([non_fraud, fraud]).sample(frac=1).reset_index(drop=True)

new_data.describe()
null_count = new_data.isnull().sum(axis=0).sort_values(ascending=False)

null_count.head(30)
values_count = new_data.nunique().sort_values()

np.sum(values_count == 1)
duplicates = []

for i, ref in enumerate(new_data.columns[:-1]):

    for other in new_data.columns[i + 1:-1]:

        if other not in duplicates and np.all(new_data[ref] == new_data[other]):

            duplicates.append(other)    

len(duplicates)
corrmat = new_data.corr()

corrmat_orig = data.corr()

f, ax = plt.subplots(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.title('Correlation matrix of sub-sampled data')

sns.heatmap(corrmat, vmax=1, square=True)

plt.subplot(1, 2, 2)

plt.title('Correlation matrix of original data')

sns.heatmap(corrmat_orig, vmax=1, square=True)
plt.figure(figsize=(16,8))

plt.subplot(1, 2, 1)

plt.title('Histogram of Time for non-fraudulent samples')

sns.distplot(non_fraud["Time"])

plt.subplot(1, 2, 2)

plt.title('Histogram of Time for fraudulent samples')

sns.distplot(fraud["Time"])
plt.figure(figsize=(16,8))

plt.subplot(1, 2, 1)

plt.title('Histogram of Time for non-fraudulent samples, mean = %f' % (non_fraud["Amount"].mean()))

sns.distplot(non_fraud["Amount"])

plt.subplot(1, 2, 2)

plt.title('Histogram of Time for fraudulent samples, mean = %f' % (fraud["Amount"].mean()))

sns.distplot(fraud["Amount"])
important_feats = new_data.columns[np.abs(corrmat["Class"]) > 0.5]

important_feats
f, ax = plt.subplots(figsize=(24, 32))

for i in range(len(important_feats) - 1):

    plt.subplot(3, 4, i + 1)

    plt.title(important_feats[i])

    sns.distplot(new_data[important_feats[i]])
f, ax = plt.subplots(figsize=(24, 32))

for i in range(len(important_feats) - 1):

    plt.subplot(3, 4, i + 1)

    plt.title(important_feats[i])

    sns.boxplot(x='Class', y=important_feats[i], data=new_data)
from sklearn.manifold import TSNE

from sklearn.manifold import SpectralEmbedding

from sklearn.decomposition import PCA, KernelPCA, FastICA
lb = new_data.quantile(0.1)

ub = new_data.quantile(0.9)

rang = ub - lb

reduced_data = new_data[~((new_data < (lb - 2 * rang)) |(new_data > (ub + 2 * rang))).any(axis=1)]

features = reduced_data.drop(['Class'], axis=1, inplace=False)

features = (features - np.mean(features)) / (np.std(features) + 1e-8)

labels = reduced_data['Class']
pca_embedding =  PCA(n_components=2) 

pca_emb_data = pca_embedding.fit_transform(features.values)

plt.figure(figsize=(10,10))

plt.scatter(pca_emb_data[labels == 1, 0], pca_emb_data[labels == 1, 1], color='red', label='positive samples')

plt.scatter(pca_emb_data[labels == 0, 0], pca_emb_data[labels == 0, 1], color='blue', label='negative samples')

plt.legend()
kpca_embedding =  KernelPCA(n_components=2, kernel='rbf')

kpca_emb_data = kpca_embedding.fit_transform(features.values)

plt.figure(figsize=(10,10))

plt.title('Reduced data with kernel PCA (RBF kernel)')

plt.scatter(kpca_emb_data[labels == 1, 0], kpca_emb_data[labels == 1, 1], color='red', label='positive samples')

plt.scatter(kpca_emb_data[labels == 0, 0], kpca_emb_data[labels == 0, 1], color='blue', label='negative samples')

plt.legend()
ica_embedding =  FastICA(n_components=2) 

ica_emb_data = ica_embedding.fit_transform(features.values)

plt.figure(figsize=(10,10))

plt.scatter(ica_emb_data[labels == 1, 0], ica_emb_data[labels == 1, 1], color='red', label='positive samples')

plt.scatter(ica_emb_data[labels == 0, 0], ica_emb_data[labels == 0, 1], color='blue', label='negative samples')

plt.legend()
tsne_embedding =  TSNE(n_components=2) 

tsne_emb_data = tsne_embedding.fit_transform(features.values)

plt.figure(figsize=(10,10))

plt.title('Reduced data with tSNE')

plt.scatter(tsne_emb_data[labels == 1, 0], tsne_emb_data[labels == 1, 1], color='red', label='positive samples')

plt.scatter(tsne_emb_data[labels == 0, 0], tsne_emb_data[labels == 0, 1], color='blue', label='negative samples')

plt.legend()
spec_embedding = SpectralEmbedding(n_components=2, affinity='rbf')

transformed_data2 = spec_embedding.fit_transform(features.values)

fig = plt.figure(figsize=(8,24))

plt.subplot(3, 1, 1)

plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')

plt.legend()

plt.subplot(3, 1, 2)

plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')

plt.legend()

plt.subplot(3, 1, 3)

plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')

plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')

plt.legend()
spec_embedding2 = SpectralEmbedding(n_components=2, affinity='nearest_neighbors', n_neighbors=30)

transformed_data2 = spec_embedding2.fit_transform(features.values)

fig = plt.figure(figsize=(8,24))

plt.subplot(3, 1, 1)

plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')

plt.legend()

plt.subplot(3, 1, 2)

plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')

plt.legend()

plt.subplot(3, 1, 3)

plt.scatter(transformed_data2[labels == 1, 0], transformed_data2[labels == 1, 1], color='red', label='positive samples')

plt.scatter(transformed_data2[labels == 0, 0], transformed_data2[labels == 0, 1], color='blue', label='negative samples')

plt.legend()