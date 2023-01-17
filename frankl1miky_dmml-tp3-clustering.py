# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv("../input/pokemon_alopez247.csv")
pokemon.head()
# Unnecessary columns
# Number and Name are just identifiers
# Total is a aggregation of others columns
clean_pokemon = pokemon.drop(columns=['Number', 'Name', 'Total'])
# missing values
null_columns = clean_pokemon.columns[clean_pokemon.isnull().any()]
null_columns
# agregate Pr_Male and hasGender by a new column named gender
# gender = 0 for female, 1 for male and 2 for other
int2gender = {
    "0": "Female",
    "1": "Male",
    "2": "Other",
}
gender = np.zeros(clean_pokemon.shape[0])
gender[clean_pokemon.Pr_Male.isnull()] = 2
gender[clean_pokemon.Pr_Male > 0.5] = 1
gender[clean_pokemon.Pr_Male <= 0.5 ] = 0
clean_pokemon['Gender'] = gender
pokemon['Gender'] = gender
print(clean_pokemon[clean_pokemon.Pr_Male.isnull() & clean_pokemon.hasGender==True].shape[0])
clean_pokemon.drop(columns=['Pr_Male', 'hasGender'], inplace=True)
# we drop columns Type_2 and Egg_Group_2 for the moment
clean_pokemon.drop(columns=["Type_2", "Egg_Group_2"], inplace=True)
for c in clean_pokemon.columns:
    if str(clean_pokemon[c].dtype).startswith("int") or str(clean_pokemon[c].dtype).startswith("float") :
        mini, maxi = clean_pokemon[c].min(), clean_pokemon[c].max()
        clean_pokemon[c] = (clean_pokemon[c] - mini) / (maxi - mini)
clean_pokemon.head()
clean_pokemon_dummies = pd.get_dummies(clean_pokemon, columns=["Type_1", "isLegendary", "Color", "Egg_Group_1", "hasMegaEvolution", "Body_Style", "Gender"])
clean_pokemon.describe()
data = clean_pokemon_dummies.as_matrix()
pca = PCA(n_components=2).fit(data)
data2d = pca.transform(data)
plt.figure(figsize=(16, 8))
scores, n_clusters, preds = [], [], []
for i in range(2, 10):
    kmean = KMeans(n_clusters = i).fit(data)
    scores.append(kmean.score(data))
    n_clusters.append(i)
    pred = kmean.predict(data)
    preds.append(pred)
    plt.subplot(2, 4, i - 1)
    plt.title(f"{i} clusters silhoute={np.round(silhouette_score(data, pred), decimals=5)}")
    plt.scatter(data2d[:, 0], data2d[:, 1], c=pred)
    
    centroids = kmean.cluster_centers_
    centroids2d = pca.transform(centroids)
    plt.plot(centroids2d[:, 0], centroids2d[:, 1], 'b+', markersize=15)
print("<< class 0 >>")
print(clean_pokemon[preds[0] == 0]['isLegendary'].value_counts())
print("<< class 1 >>")
print(clean_pokemon[preds[0] == 1]['isLegendary'].value_counts(), end="\n")
print("<< class 0 >>")
print(clean_pokemon[preds[0] == 0]['hasMegaEvolution'].value_counts())
print("<< class 1 >>")
print(clean_pokemon[preds[0] == 1]['hasMegaEvolution'].value_counts(), end="\n")
print(int2gender)
print("<< class 0 >>")
print(pokemon[preds[1] == 0]['Gender'].value_counts())
print("<< class 1 >>")
print(pokemon[preds[1] == 1]['Gender'].value_counts(), end="\n")
print("<< class 2 >>")
print(pokemon[preds[1] == 2]['Gender'].value_counts(), end="\n")
plt.plot(n_clusters, -np.array(scores))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.figure(figsize=(16, 8))
scores, n_clusters, preds = [], [], []
for i in range(2, 10):
    agglo = AgglomerativeClustering(n_clusters = i)
#     scores.append(agglo.score(data))
    n_clusters.append(i)
    pred = agglo.fit_predict(data)
    preds.append(pred)
    plt.subplot(2, 4, i - 1)
    plt.title(f"Agglo - {i} clusters silhoute={np.round(silhouette_score(data, pred), decimals=5)}")
    plt.scatter(data2d[:, 0], data2d[:, 1], c=pred)
plt.figure(figsize=(16, 8))
for i in range(10, 90, 10):
    dbscan = DBSCAN(eps=i, min_samples=5)
    pred = dbscan.fit_predict(data)
    plt.subplot(2, 4, i/10)
    plt.title(f"DBSCAN - {i} eps")
    plt.scatter(data2d[:, 0], data2d[:, 1], c=pred)