import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid", {'axes.grid' : False})
raw_data = pd.read_csv('../input/NBALeagueAverages.csv')

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

df = raw_data.copy()

df
plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['PTS'], label="Points per game", marker='o')



plt.title('League Average in Points per game')

plt.xlabel('Season')

plt.ylabel("Points")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()
plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['DRB'], label="Defensive Rebounds", marker='o')

plt.plot(df['Season'],df['ORB'], label="Offensive Rebounds", marker='o')

plt.plot(df['Season'],df['TRB'], label="Total Rebounds", marker='o')



plt.title('League Average in Rebounds per game')

plt.xlabel('Season')

plt.ylabel("Rebounds")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()

plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['AST'], label="Assists", marker='o')

plt.plot(df['Season'],df['TOV'], label="Turnovers", marker='o')



plt.title('League Average in Assists and Turnovers per game')

plt.xlabel('Season')

plt.ylabel("Assists/Turnovers")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()
plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['BLK'], label="Blocks", marker='o')

plt.plot(df['Season'],df['STL'], label="Steals", marker='o')



plt.title('League Average in Blocks and Steals per game')

plt.xlabel('Season')

plt.ylabel("Blocks/Steals")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()
plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['eFG%'], label="Effective Field Goal Percentage", marker='o')

plt.plot(df['Season'],df['TS%'], label="True Shooting Percentage", marker='o')

plt.plot(df['Season'],df['FTA/FGA'], label="Free Throw Rate", marker='o')

plt.plot(df['Season'],df['3PA/FGA'], label="Three Point Rate", marker='o')



plt.title('League Average in Advanced Shooting stats')

plt.xlabel('Season')

plt.ylabel("%")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()
plt.figure(figsize=(12,8))



plt.plot(df['Season'],df['Pace'], label="Pace", marker='o')



plt.title('League Average in Pace')

plt.xlabel('Season')

plt.ylabel("Pace")

plt.xticks(rotation=90)

plt.gca().invert_xaxis()

plt.legend()

plt.show()
df_stat = df.drop(['Index', 'Season'],axis=1)

df_stat.head()
scaler = StandardScaler()

df_std = scaler.fit_transform(df_stat)

df_std
pca = PCA()

pca.fit(df_std)
pca.explained_variance_ratio_
plt.figure(figsize = (12,9))

plt.plot(range(1,31), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')

plt.title('Explained Variance by Components')

plt.xlabel('Number of Components')

plt.ylabel('Cumulative Explained Variance')
pca = PCA(n_components = 4)
pca.fit(df_std)
pca.components_
df_pca_comp = pd.DataFrame(data = pca.components_,

                           columns = df_stat.columns.values,

                           index = ['Component 1', 'Component 2', 'Component 3', 'Component 4'])

df_pca_comp
pca.transform(df_std)
scores_pca = pca.transform(df_std)
wcss = []

for i in range(1,11):

    kmeans_pca = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans_pca.fit(scores_pca)

    wcss.append(kmeans_pca.inertia_)

    

plt.figure(figsize = (10,8))

plt.plot(range(1, 11), wcss, marker = 'o', linestyle = '--')

plt.xlabel('Number of Clusters')

plt.ylabel('WCSS')

plt.title('K-means with PCA Clustering')

plt.show()
kmeans_pca = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

kmeans_pca.fit(scores_pca)
df_segm_pca_kmeans = pd.concat([df_stat.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)

df_segm_pca_kmeans.columns.values[-4: ] = ['Component 1', 'Component 2', 'Component 3', 'Component 4']

df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_
df_segm_pca_kmeans
df_segm_pca_kmeans_grouped = df_segm_pca_kmeans.groupby(['Segment K-means PCA']).mean()

df_segm_pca_kmeans_grouped
df_segm_pca_kmeans_renamed = df_segm_pca_kmeans_grouped.rename({0:'Defense Heavy Era', 

                                                          1:'Paint Centered Era',

                                                          2:'Analytics Era', 

                                                          3:'Balanced Era'})

df_segm_pca_kmeans_renamed
df_segm_pca_kmeans['Legend'] = df_segm_pca_kmeans['Segment K-means PCA'].map({0:'Defense Heavy Era', 

                                                          1:'Paint Centered Era',

                                                          2:'Analytics Era', 

                                                          3:'Balanced Era'})
x_axis = df_segm_pca_kmeans['Component 1']

y_axis = df_segm_pca_kmeans['Component 2']

plt.figure(figsize = (10, 8))

sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans['Legend'], palette = ['g', 'r', 'c', 'm'])

plt.title('2D Represntation of Clusters by PCA Components')

plt.show()
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')



x0 = df_segm_pca_kmeans['Component 1'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 2]

y0 = df_segm_pca_kmeans['Component 2'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 2]

z0 = df_segm_pca_kmeans['Component 3'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 2]

x1 = df_segm_pca_kmeans['Component 1'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 1]

y1 = df_segm_pca_kmeans['Component 2'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 1]

z1 = df_segm_pca_kmeans['Component 3'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 1]

x2 = df_segm_pca_kmeans['Component 1'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 3]

y2 = df_segm_pca_kmeans['Component 2'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 3]

z2 = df_segm_pca_kmeans['Component 3'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 3]

x3 = df_segm_pca_kmeans['Component 1'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 0]

y3 = df_segm_pca_kmeans['Component 2'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 0]

z3 = df_segm_pca_kmeans['Component 3'].loc[df_segm_pca_kmeans['Segment K-means PCA'] == 0]



ax.scatter(x0, y0, z0, c='g', marker='o', label='Analytics Era')

ax.scatter(x1, y1, z1, c='r', marker='o', label='Balanced Era')

ax.scatter(x2, y2, z2, c='c', marker='o', label='Defense Heavy Era')

ax.scatter(x3, y3, z3, c='m', marker='o', label='Paint Centered Era')

ax.set_xlabel('Component 1')

ax.set_ylabel('Component 2')

ax.set_zlabel('Component 3')

ax.set_title('3D Representation of Clusters by PCA Components')

ax.legend()

plt.show()

df_season = df.iloc[:, 1:2]

df_merged = df_season.join(df_segm_pca_kmeans, how='outer')

#pd.set_option('display.max_columns', None)

df_merged