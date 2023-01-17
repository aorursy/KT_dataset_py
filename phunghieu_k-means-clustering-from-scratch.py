import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
DATA_PATH = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'

K = 5
df = pd.read_csv(DATA_PATH)
df
df.Gender.replace('Male', 0, inplace=True)

df.Gender.replace('Female', 1, inplace=True)
df
fig, axes = plt.subplots(2, 2, figsize=(24, 8))



sns.distplot(df.Gender, ax=axes[0, 0])

sns.distplot(df.Age, ax=axes[0, 1])

sns.distplot(df['Annual Income (k$)'], ax=axes[1, 0])

sns.distplot(df['Spending Score (1-100)'], ax=axes[1, 1])

plt.show()
fig, ax = plt.subplots(figsize=(16, 8))



sns.scatterplot(df['Annual Income (k$)'], df['Spending Score (1-100)'], ax=ax)

plt.show()
class KMeans(object):

    def __init__(self, k=5):

        self.k = k

        

        

    def _init_centroids(self, df):

        centroids = df.sample(self.k)

        

        return centroids

    

    @staticmethod

    def _compute_distance(x1, x2):

        return np.sum((x1 - x2)**2)**0.5

    

    def _cluster(self, df, centroids):

        cluster_col = []

        for idx, item in df.iterrows():

            distances = [self._compute_distance(item.to_numpy(), centroid.to_numpy()) for _, centroid in centroids.iterrows()]

            cluster_col.append(np.argmin(distances))

            

        df['cluster'] = cluster_col

        

        return df

    

    def _compute_centroids(self, df, feature_cols):

        centroids = []

        for cluster_id in range(self.k):

            cluster = df.loc[df.cluster == cluster_id, feature_cols]

            centroids.append(cluster.mean())

    

        return pd.DataFrame(centroids)

    

    @staticmethod

    def _centroids_diff(old_centroids, new_centroids, epsilon=1e-7):

        for old_centroid, new_centroid in zip(old_centroids.to_numpy(), new_centroids.to_numpy()):

            for old_val, new_val in zip(old_centroid, new_centroid):

                if np.abs(old_val - new_val) > epsilon:

                    return True

            

        return False

            

        

    def fit(self, df, feature_cols=['Annual Income (k$)', 'Spending Score (1-100)']):

        centroids = self._init_centroids(df.loc[:, feature_cols])

        

        df = self._cluster(df.loc[:, feature_cols], centroids)

        

        new_centroids = self._compute_centroids(df, feature_cols)

        

        while self._centroids_diff(centroids, new_centroids):

            centroids = new_centroids

            df = self._cluster(df.loc[:, feature_cols], centroids)

            new_centroids = self._compute_centroids(df, feature_cols)

        

        return df, centroids
model = KMeans()
new_df, centroids = model.fit(df)
fig, ax = plt.subplots(figsize=(16, 8))



sns.scatterplot(data=new_df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', palette='pastel', ax=ax)

plt.scatter(centroids['Annual Income (k$)'], centroids['Spending Score (1-100)'], linewidths=5, c='k')

plt.show()