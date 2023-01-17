import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Mini Batch K-Means used because less sensitive to outliers (Laos Dataset has many outliers).

from sklearn.cluster import MiniBatchKMeans
mb_kmeans = MiniBatchKMeans(n_clusters=3, n_init=100, random_state=123)
#Dataset from last population census in Laos 2015. Public accessible via decide.la

#Subset: only variables from following dimensions Education, Health, Economic activites, Poverty, Living conditions.

subset = pd.read_excel("../input/laos-population-census-2015/Laos_Population_Census_2015.xls", sheet_name="subset", index_col=0)
subset.head()
#Standardize attributes to make scales comparable

X = (subset - subset.mean(axis=0)) / subset.std(axis=0)
#apply mini batch k-means algorithm

mb_kmeans.fit(X)
#Objective function value with 3 clusters

mb_kmeans.inertia_
#use elbow heuristic

num_clusters = list(range(1,20))

obv = []

for k in num_clusters:

    model = MiniBatchKMeans(n_clusters=k, n_init=100)

    model.fit(X)

    obv.append(model.inertia_)
#plot elbow

plt.plot(num_clusters, obv, "-o")

plt.xlabel("Number of Clusters (k)")

plt.ylabel("OBV")
#change parameter to n_clusters=4

mb_kmeans = MiniBatchKMeans(n_clusters=4, n_init=100, random_state=123)
#apply mini batch k-means algorithm

mb_kmeans.fit(X)
#insert labels to original dataframe

subset["clusters"] = mb_kmeans.labels_
#get a feeling for the clusters

subset.groupby("clusters").size()
subset.groupby("clusters").median()
#replace integers with self-defined labels

subset.replace([0, 1, 2, 3], ["poor", "rich", "middle class", "indefinable"], inplace=True)
#import packages for map visualization

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon
#Import Laos Shapefile 

laos = gpd.read_file("../input/laos-shapefiles/laos_area.shp")[["uuid", "geometry"]]
#merge geodataset "laos" with "subset"

merged_subset = laos.merge(subset, left_on='uuid', right_on='uuid')
fig, ax = plt.subplots(figsize = (15,15))

merged_subset.plot(column="clusters", ax = ax, cmap="RdBu", linewidth=0.1 , legend=True)

ax.axis("off")

#plt.savefig('poverty_categorization.png', dpi=300)