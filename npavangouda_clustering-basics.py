# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.manifold import Isomap

from sklearn.manifold import TSNE



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_wine = pd.read_csv('../input/Wine.csv')

df_wine.describe()
df_wine.head(5)
df_wine.info()
df_wine['Customer_Segment'].unique()
# seperate the label and featuers

X = df_wine.drop('Customer_Segment', axis=1)

y = df_wine['Customer_Segment']
wine_correlation = X.corr()

plt.figure(figsize=(25,25))

sns.heatmap(wine_correlation, annot=True, cmap ="RdBu_r");
# Based on heat map Flavanoids is linearly related to Total_Phenols

sns.scatterplot(x='Flavanoids', y='Total_Phenols', data=df_wine);
# Based on heat map Flavanoids is linearly related to OD280

sns.scatterplot(x='Flavanoids', y='OD280', data=df_wine);
pca_wine = PCA(n_components=2)



principalComp_wine = pca_wine.fit_transform(X)

print('Explained variation per principal component: {}'.format(pca_wine.explained_variance_ratio_))
principal_wine_Df = pd.DataFrame(data = principalComp_wine, columns = ['principal component 1', 'principal component 2'])

principal_wine_Df.head(5)
plt.figure(figsize=(10,8))

plt.scatter(principal_wine_Df['principal component 1'],principal_wine_Df['principal component 2'], c=y);
embedding = Isomap(n_components=2,n_neighbors=40)

X_isomap = embedding.fit_transform(X)



plt.figure(figsize=(10, 8))

plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y);
tsneModel = TSNE(n_components=2, random_state=0, perplexity=50, n_iter=5000)

tsne_wine_data =  tsneModel.fit_transform(X)



tsne_wine_data = np.vstack((tsne_wine_data.T, y)).T



tsne_wine_df = pd.DataFrame(data=tsne_wine_data, columns=('Dim_1', 'Dime_2', 'label'))



sns.FacetGrid(tsne_wine_df, hue='label', size=6).map(plt.scatter,  'Dim_1', 'Dime_2')

plt.show();
X.head()
from sklearn import preprocessing
minmaxScaler = preprocessing.MinMaxScaler()

X_scaled_df = minmaxScaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled_df, columns=X.columns)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,5))

ax1.set_title('Before Scaling')

sns.kdeplot(X['Alcohol'], ax=ax1)

sns.kdeplot(X['Magnesium'], ax=ax1)

sns.kdeplot(X['Proline'], ax=ax1)



ax2.set_title('After Min-Max Scaling')

sns.kdeplot(X_scaled_df['Alcohol'], ax=ax2)

sns.kdeplot(X_scaled_df['Magnesium'], ax=ax2)

sns.kdeplot(X_scaled_df['Proline'], ax=ax2)



plt.show();
from sklearn import metrics

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
X_scaled_df.shape
X_scaled_df.describe()
sScores = []

for n_clusters in range(2,30):

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)

    kmeans.fit(X_scaled_df)

    clusters = kmeans.predict(X_scaled_df)

    silhouette_avg = silhouette_score(X_scaled_df, clusters)

    sScores.append(silhouette_avg)



silhouette_df = pd.DataFrame({'No_Clusters':range(2,30), 'sScore':sScores})
sns.set()

plt.figure(figsize=(10, 6))

ax = sns.lineplot(x='No_Clusters', y='sScore', data=silhouette_df);

ax.set(xticks=silhouette_df['No_Clusters']);

# from the plot below we can conclude that 3 is a good cluster number as Silhouette Score is the highest.
wss = []

for n_clusters in range(2,30):

    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)

    kmeans.fit(X_scaled_df)

    wss.append(kmeans.inertia_)

    



elbow_df = pd.DataFrame({'No_Clusters':range(2,30), 'sScore':wss})

    
plt.figure(figsize=(10, 6))

ax = sns.lineplot(x='No_Clusters', y='sScore', data=elbow_df);

ax.set(xticks=elbow_df['No_Clusters']);

# from the plot below we can conclude that 3 is a good cluster number as the  elbow is at 3.
# as per the elbow-method or Silhouette plot the number of clusters is => 3

kmeans_tsne_2d_df = tsne_wine_df.drop('label', axis=1)

kmeans_tsne = KMeans(n_clusters=3, random_state=0)

kmeans_tsne_labels = kmeans_tsne.fit_predict(kmeans_tsne_2d_df)

kmeans_tsne_centers = kmeans_tsne.cluster_centers_

kmeans_tsne_2d_df['label'] = kmeans_tsne_labels.tolist()
kmeans_tsne_2d_df.head()
from scipy.spatial.distance import cdist



fig, ax = plt.subplots(figsize=(10, 10))

ax.axis('equal')

sns.scatterplot(x='Dim_1', y='Dime_2', hue='label', data=kmeans_tsne_2d_df, palette=['red','blue','green'], ax=ax)

plt.scatter(kmeans_tsne_centers[:,0], kmeans_tsne_centers[:, 1], s = 300, c = 'black' , label = 'centeroid', marker='*')

radii = [cdist(kmeans_tsne_2d_df[kmeans_tsne_labels == i].drop('label', axis=1), [center]).max() for i, center in enumerate(kmeans_tsne_centers)]

for c, r in zip(kmeans_tsne_centers, radii):

    ax.add_patch(plt.Circle(c, r, alpha=0.3))

        

plt.show();
from sklearn.mixture import GaussianMixture



gmm_tsne_2d_df = tsne_wine_df.drop('label', axis=1)



gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')

gmm_tsne_labels = gmm.fit(gmm_tsne_2d_df).predict(gmm_tsne_2d_df)

gmm_tsne_centers = gmm.means_

gmm_tsne_2d_df['label'] = gmm_tsne_labels.tolist()
gmm_tsne_2d_df.head()
from matplotlib.patches import Ellipse



fig, ax = plt.subplots(figsize=(10, 10))

ax.axis('equal')

sns.scatterplot(x='Dim_1', y='Dime_2', hue='label', data=gmm_tsne_2d_df, palette=['red','blue','green'], ax=ax)



plt.scatter(gmm_tsne_centers[:,0], gmm_tsne_centers[:, 1], s = 300, c = 'black' , label = 'centeroid', marker='*')

ax.add_patch(Ellipse(gmm_tsne_centers[0], 2, 5.5, 150, alpha=0.3));

ax.add_patch(Ellipse(gmm_tsne_centers[1], 2, 6.6, 145, alpha=0.3));

ax.add_patch(Ellipse(gmm_tsne_centers[2], 2, 7, 145, alpha=0.3));



plt.show();