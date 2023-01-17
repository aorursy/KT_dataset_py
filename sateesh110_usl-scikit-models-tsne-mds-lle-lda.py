### DataFrame ####

import numpy as np

import pandas as pd



### Visualization ####

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from matplotlib import offsetbox



### Scikit ####

from sklearn.model_selection import train_test_split

np.random.seed(42)

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale, StandardScaler

from sklearn.decomposition import PCA

from sklearn.datasets import fetch_openml

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE

from sklearn import (manifold, datasets, decomposition, ensemble, 

                     discriminant_analysis, random_projection, neighbors)



### Others ###

import time

import warnings

warnings.filterwarnings('ignore')

from time import time

# for kaggle kernel

train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")
#train_set = pd.read_csv("train.csv")

#test_set = pd.read_csv("test.csv")
# Create a copy

train = train_set.copy()

test = test_set.copy()
# For reproducability of the results

np.random.seed(42)

rndperm_train = np.random.permutation(train.shape[0])

rndperm_test = np.random.permutation(test.shape[0])
# get columns

feat_cols = train.drop(columns=['label']).columns

len(feat_cols)
# Plot graph of each digit



plt.gray()

fig = plt.figure( figsize=(16,7) )

for i in range(0,15):

    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(train.loc[rndperm_train[i],'label'])) )

    ax.matshow(train.loc[rndperm_train[i],feat_cols].values.reshape((28,28)).astype(float))

plt.show()
# Random instance of N

N = 10000

df_subset_train = train.loc[rndperm_train[:N],:].copy()

df_subset_test = test.loc[rndperm_test[:N],:].copy()
# Check for nulls

df_subset_train['label'].isnull().sum()
# drop nulls

df_subset_train = df_subset_train.dropna(how='any')
# recheck for nulls

df_subset_train['label'].isnull().sum()
# Subset shape

df_subset_train.shape
# Normalising data by dividing it by 255 should improve activation functions performance

y_train = df_subset_train['label'].values

X_train = df_subset_train.drop(columns=['label']).values/255
X_train.shape
y_train.shape
# get columns

feat_cols = df_subset_train.drop(columns=['label']).columns

len(feat_cols)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

#X_test = scaler.transform(X_test)
pca = PCA()

pca.fit(X_train)
# PCA features

features = range(pca.n_components_)

features
# number of intrinsic dimensions

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95)+1

d
# total variance of the intrinsic dimensions

np.sum(pca.explained_variance_ratio_)
# datasets after dimensionality reduction



pca = PCA(n_components=d)

pca.fit(X_train)

X_train_reduced = pca.transform(X_train)

#X_test_reduced = pca.transform(X_test)



print(X_train_reduced.shape)

#print(X_test_reduced.shape)
df_subset_train['pca-one'] = X_train_reduced[:,0]

df_subset_train['pca-two'] = X_train_reduced[:,1] 
plt.figure(figsize=(16,10))

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)

tsne_pca_results = tsne.fit_transform(X_train_reduced)
df_subset_train['tsne-2d-one'] = tsne_pca_results[:,0]

df_subset_train['tsne-2d-two'] = tsne_pca_results[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
plt.figure(figsize=(16,7))

ax1 = plt.subplot(1, 2, 1)

sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3,

    ax=ax1

)

ax2 = plt.subplot(1, 2, 2)

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3,

    ax=ax2

)
# Random 2D projection using a random unitary matrix

print("Computing random projection")

rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)

X_projected = rp.fit_transform(X_train_reduced)



df_subset_train['random-2d-one'] = X_projected[:,0]

df_subset_train['random-2d-two'] = X_projected[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="random-2d-one", y="random-2d-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)

# Projection on to the first 2 linear discriminant components



print("Computing Linear Discriminant Analysis projection")

X2 = X_train_reduced.copy()

X2.flat[::X_train_reduced.shape[1] + 1] += 0.01  # Make X_train_reduced invertible



X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y_train)
df_subset_train['LDA-one'] = X_lda[:,0]

df_subset_train['LDA-two'] = X_lda[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="LDA-one", y="LDA-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)

# Isomap projection of the digits dataset

print("Computing Isomap projection")

n_neighbors = 30

X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X_train_reduced)
df_subset_train['ISO-one'] = X_iso[:,0]

df_subset_train['ISO-two'] = X_iso[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="ISO-one", y="ISO-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
# Locally linear embedding of the digits dataset

print("Computing LLE embedding")

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,

                                      method='standard')

X_lle = clf.fit_transform(X_train_reduced)

print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
df_subset_train['LLE_std-one'] = X_lle[:,0]

df_subset_train['LLE_std-two'] = X_lle[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="LLE_std-one", y="LLE_std-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
# Modified Locally linear embedding of the digits dataset

print("Computing modified LLE embedding")

clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,

                                      method='modified')

t0 = time()

X_mlle = clf.fit_transform(X_train_reduced)

print("Done. Reconstruction error: %g" % clf.reconstruction_error_)

df_subset_train['LLE_mod-one'] = X_mlle[:,0]

df_subset_train['LLE_mod-two'] = X_mlle[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="LLE_mod-one", y="LLE_mod-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
# MDS  embedding of the digits dataset

print("Computing MDS embedding")

clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)

t0 = time()

X_mds = clf.fit_transform(X_train_reduced)

print("Done. Stress: %f" % clf.stress_)
df_subset_train['MDS-one'] = X_mds[:,0]

df_subset_train['MDS-two'] = X_mds[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="MDS-one", y="MDS-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)
# Random Trees embedding of the digits dataset

print("Computing Totally Random Trees embedding")

hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,

                                       max_depth=5)



X_transformed = hasher.fit_transform(X_train)

pca = decomposition.TruncatedSVD(n_components=2)

X_rte_reduced = pca.fit_transform(X_transformed)
df_subset_train['RTE-one'] = X_rte_reduced[:,0]

df_subset_train['RTE-two'] = X_rte_reduced[:,1]

plt.figure(figsize=(16,10))

sns.scatterplot(

    x="RTE-one", y="RTE-two",

    hue="label",

    palette=sns.color_palette("hls", 10),

    data=df_subset_train,

    legend="full",

    alpha=0.3

)