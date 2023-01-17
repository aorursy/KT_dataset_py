import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

np.random.seed(200)
# Initializing the values

df = pd.DataFrame({

    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],

    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]

})



df.head()
#let's select 3 centroids or cluster centers randomnly

k = 3

#centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(0, 80), np.random.randint(0, 80)] for i in range(k)

}

print(centroids)



fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['x'], df['y'], color='k')

colmap = {1: 'r', 2: 'g', 3: 'b'}

for i in centroids.keys():

    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)

# plt.show()
#Calculating the distance and assigning the clusters to the datapoints

def assignment(df, centroids):

    for i in centroids.keys():

        # sqrt((x1 - x2)^2 - (y1 - y2)^2)

        df['distance_from_{}'.format(i)] = (

            np.sqrt((df['x'] - centroids[i][0]) ** 2 + (df['y'] - centroids[i][1]) ** 2)

        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df



df = assignment(df, centroids)

print(df.head())



fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)

# plt.show()
# Update Stage

import copy



old_centroids = copy.deepcopy(centroids)



def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])

        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])

    return k



centroids = update(centroids)

    

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)



for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])

# plt.show()
# Repeat Assigment Stage



df = assignment(df, centroids)



# Plot results

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)

# plt.show()
# Continue until all assigned categories don't change any more

while True:

    closest_centroids = df['closest'].copy(deep=True)

    centroids = update(centroids)

    df = assignment(df, centroids)

    if closest_centroids.equals(df['closest']):

        break



fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    ax.scatter(*centroids[i], color=colmap[i])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)

# plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

np.random.seed(200)

from sklearn.cluster import KMeans
df = pd.DataFrame({

    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],

    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]

})



kmeans = KMeans(n_clusters=2)  #here n_clusters is selected as 2 randomly.

kmeans.fit(df)
#print SSE

kmeans.inertia_
labels = kmeans.predict(df)

labels
centroids = kmeans.cluster_centers_

centroids
fig, ax = plt.subplots(figsize=(5, 5))

colmap = {1: 'r', 2: 'g', 3: 'b'}

colors = map(lambda x: colmap[x+1], labels)

colors1=list(colors)

print(colors1)



ax.scatter(df['x'], df['y'], color=colors1, alpha=0.5, edgecolor='k')

for idx, centroid in enumerate(centroids):

    ax.scatter(*centroid, color=colmap[idx+1])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)
# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []

for cluster in range(1,20):

    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')

    kmeans.fit(df)

    SSE.append(kmeans.inertia_)



# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})

fig, ax = plt.subplots(figsize=(12,6))

ax.plot(frame['Cluster'], frame['SSE'], marker='o')

ax.set_xlabel('Number of clusters')

ax.set_ylabel('Inertia')
# k means using 3 clusters and k-means++ initialization

kmeans = KMeans(n_jobs = -1, n_clusters = 3, init='k-means++')

kmeans.fit(df)



print(kmeans.inertia_)

pred = kmeans.predict(df)

print(pred)



fig, ax = plt.subplots(figsize=(5, 5))

colmap = {1: 'r', 2: 'g', 3: 'b'}

colors = map(lambda x: colmap[x+1], labels)

colors1=list(colors)

print(colors1)



ax.scatter(df['x'], df['y'], color=colors1, alpha=0.5, edgecolor='k')

for idx, centroid in enumerate(kmeans.cluster_centers_):

    ax.scatter(*centroid, color=colmap[idx+1])

ax.set_xlim(0, 80)

ax.set_ylim(0, 80)