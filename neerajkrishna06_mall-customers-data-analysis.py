import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
mall_data = pd.read_csv("../input/Mall_Customers.csv")
mall_data.head()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')

sns.countplot(mall_data["Gender"])
fig = plt.figure(figsize=(15, 6))

n = 0

for i in mall_data.columns[2:]:

    n+=1

    sns.set_style('darkgrid')

    ax = fig.add_subplot(1, 3, n)

    sns.distplot(mall_data[i], ax=ax)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    ax.set_title("Distribution of {}".format(i))
fig = plt.figure(figsize=(15, 6))

n = 0

for i in mall_data.columns[2:]:

    n+=1

    sns.set_style('darkgrid')

    ax = fig.add_subplot(1, 3, n)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    sns.violinplot(x=i, y="Gender", data=mall_data, palette='vlag')

    sns.swarmplot(x=i, y="Gender", data=mall_data)

    ax.set_ylabel("Gender" if n==1 else '')

    ax.set_title("Distribution wrt Gender" if n==2 else '')
plt.figure(figsize=(8, 6))

sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=mall_data, hue="Gender")
mall_data.isnull().sum()
from sklearn.cluster import KMeans



x_km = mall_data[["Annual Income (k$)", "Spending Score (1-100)"]]

km = KMeans(n_clusters=5, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

y_km = km.fit_predict(x_km)
# plotting the cluster centroids



# Clusters

md_segment = pd.concat([mall_data["Annual Income (k$)"], mall_data["Spending Score (1-100)"], 

                        pd.Series(y_km, name="Cluster Index")], axis=1)



# Plotting

markers = {0:'s', 1:'o', 2:'X', 3:'D', 4:'v'}

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1)

sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=md_segment, markers=markers,

                style = 'Cluster Index', ax=ax, s=75, hue="Cluster Index", palette='Set2')



# Cluster Centroids

cluster_centroids = km.cluster_centers_

centroid_X = [x[0] for x in cluster_centroids]

centroid_Y = [x[1] for x in cluster_centroids]



ax.scatter(centroid_X, centroid_Y, color='red', marker='*', s=75, label="Centroids")

plt.legend(loc=0)
# Spending Score vs Age

plt.figure(figsize=(8, 6))

sns.scatterplot(x="Age", y="Spending Score (1-100)", data=mall_data)
# Elbow Method



x1_km = mall_data[["Age", "Spending Score (1-100)"]]

distortions = []

for i in range(1, 11):

    km1 = KMeans(n_clusters=i)

    km1.fit(x1_km)

    distortions.append(km1.inertia_)



# Plotting

plt.figure(figsize=(8, 6))

plt.plot(list(range(1, 11)), distortions, marker='o')

plt.xlabel("Number of Clusters - K")

plt.xticks(list(range(1, 11)))

plt.ylabel("Distortion Value")

plt.title("Elbow Method")

plt.show()
# plotting the cluster centroids



km1 = KMeans(n_clusters=4)

y1_km = km1.fit_predict(x1_km)



# Clusters

md1_segment = pd.concat([mall_data["Age"], mall_data["Spending Score (1-100)"], 

                        pd.Series(y1_km, name="Cluster Index")], axis=1)



# Plotting

markers = {0:'s', 1:'o', 2:'X', 3:'D'}

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(1, 1, 1)

sns.scatterplot(x="Age", y="Spending Score (1-100)", data=md1_segment, markers=markers,

                style = 'Cluster Index', ax=ax, s=75, hue="Cluster Index", palette='Set2')



# Cluster Centroids

cluster_centroids = km1.cluster_centers_

centroid_X = [x[0] for x in cluster_centroids]

centroid_Y = [x[1] for x in cluster_centroids]



ax.scatter(centroid_X, centroid_Y, color='red', marker='*', s=75, label="Centroids")

plt.legend(loc=0)
plt.figure(figsize=(8, 6)) 

sns.scatterplot(x="Age", y="Annual Income (k$)", data=mall_data, hue="Gender")