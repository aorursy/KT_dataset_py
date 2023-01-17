# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
anime = pd.read_csv("../input/anime.csv")
ratings = pd.read_csv("../input/rating.csv")

# Any results you write to the current directory are saved as output.
from __future__ import print_function
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
from sklearn import preprocessing
import seaborn as sns
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.neighbors import NearestNeighbors
anime.head()
genre_dummies = anime["genre"].str.get_dummies(sep=",")
anime = pd.concat([anime, genre_dummies], axis=1)
anime.drop(["genre"], axis=1, inplace=True)
anime.head()
null_columns=anime.columns[anime.isnull().any()]
anime[null_columns].isnull().sum()
anime["type"].value_counts()
anime["type"].fillna("TV", inplace=True)
grouped_type = anime.groupby(["type"])
grouped_type_median = grouped_type.median()
grouped_type_median
# helper function to find median rating of a particular anime type
def fillRatings(row, grouped_median):
    return grouped_median.loc[row["type"]]["rating"]
anime.rating = anime.apply(lambda row: fillRatings(row, grouped_type_median) if np.isnan(row['rating'])  else row["rating"], axis=1)
anime[anime["episodes"] == "Unknown"]
anime["episodes"] = anime["episodes"].apply(lambda x: 0 if x == "Unknown" else x)
le = preprocessing.LabelEncoder()
anime["type"] = le.fit_transform(list(anime["type"].values))
anime_corr_df = anime.copy(deep=True)
anime_corr_df.drop(["anime_id", "name", "members"], axis=1, inplace=True)
k = 20 #number of variables for heatmap
corr = anime_corr_df.corr()
cols = corr.nlargest(k, 'rating')['rating'].index
cm = np.corrcoef(anime_corr_df[cols].values.T)
sns.set(font_scale=1.25)
plt.figure(figsize=(12, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
def getScores(num_clusters):
    clusterer = KMeans(n_clusters=num_clusters, random_state=42).fit(anime_corr_df)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(anime_corr_df)

    # TODO: Find the cluster centers
    centers = clusterer.cluster_centers_

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(anime_corr_df,preds)
    return score

scores = pd.DataFrame(columns=['Silhouette Score'])
scores.columns.name = 'Number of Clusters'    
for i in range(2,10):
    score = getScores(i) 
    scores = scores.append(pd.DataFrame([score],columns=['Silhouette Score'],index=[i]))

display(scores)
clusterer = KMeans(n_clusters=2, random_state=42).fit(anime_corr_df)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(anime_corr_df)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(anime_corr_df,preds)

print(score)
clusters = clusterer.labels_.tolist()
animes = { 'name': np.array(anime.name), 'cluster': clusters}
frame = pd.DataFrame(animes, index = [clusters] , columns = ['name', 'cluster'])
frame['cluster'].value_counts()
def showClusters(clusterer, frame, num_clusters):
    print("Top terms per cluster:")
    print()
    #sort cluster centers by proximity to centroid
    order_centroids = clusterer.cluster_centers_.argsort()[:, ::-1] 

    for i in range(num_clusters):
        print("Cluster %d names:" % (i+1), end='')
        for title in frame.ix[i]['name'].values.tolist()[0:50]:
            print(' %s,' % title, end='')
        print() 
        print()
showClusters(clusterer, frame, 2)
res=[]
for k in range(2,20):
    kmeans = KMeans(n_clusters=k,random_state=42)
    model=kmeans.fit(anime_corr_df)
    wssse=kmeans.inertia_
    KW=(k,wssse)
    KW
    res.append(KW)
plt.plot(*zip(*res))
plt.show()
clusterer = KMeans(n_clusters=6, random_state=42).fit(anime_corr_df)

# TODO: Predict the cluster for each data point
preds = clusterer.predict(anime_corr_df)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_

clusters = clusterer.labels_.tolist()
animes = { 'name': np.array(anime.name), 'cluster': clusters}
frame = pd.DataFrame(animes, index = [clusters] , columns = ['name', 'cluster'])
frame['cluster'].value_counts()
showClusters(clusterer, frame, 6)
neighbours = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(anime_corr_df)
distances, indices = neighbours.kneighbors(anime_corr_df)
def get_index_from_name(name):
    return anime[anime["name"]==name].index.tolist()[0]
# method to find the similar animes
def print_similar_animes(query):
    anime_id = get_index_from_name(query)
    for id in indices[anime_id][1:]:
        print(anime.ix[id]["name"])
print_similar_animes("Hunter x Hunter (2011)")
print_similar_animes("Doraemon (1979)")
print_similar_animes("Naruto")