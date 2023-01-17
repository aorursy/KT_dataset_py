import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.DataFrame({

    'x': [12, 20, 28, 18, 92, 81, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 11, 90, 81 ,31, 56, 85, 71, 13, 10, 9],

    'y': [39, 36, 30, 52, 54, 46, 88, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 99, 30, 41, 68, 74, 82 ,91, 3, 55, 63]

})

k = 3

centroids = {1: [12, 36], 2: [51, 66], 3: [55, 14]}

    

fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color='k')

colmap = {1: 'r', 2: 'g', 3: 'b'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()

print(centroids)
def assignment(df, centroids):

    for i in centroids.keys():

        # sqrt((x1 - x2)^2 - (y1 - y2)^2)

        df['distance_from_{}'.format(i)] = (

            np.sqrt(

                (df['x'] - centroids[i][0]) ** 2

                + (df['y'] - centroids[i][1]) ** 2

            )

        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df



df = assignment(df, centroids)

print(df.head())





fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()
import copy





old_centroids = copy.deepcopy(centroids)



def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])

        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])

    return k



centroids = update(centroids)

    

fig = plt.figure(figsize=(5, 5))

ax = plt.axes()

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])

plt.show()
df = assignment(df, centroids)



# Plot results

fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()
df = assignment(df, centroids)



# Plot results

fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()
df = assignment(df, centroids)



fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()
df = assignment(df, centroids)



fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()





df1 = pd.DataFrame({

    'x': [12, 20, 28, 18, 92, 81, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 11, 90, 81 ,31, 56, 85, 71, 13, 10, 9],

    'y': [39, 36, 30, 52, 54, 46, 88, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 99, 30, 41, 68, 74, 82 ,91, 3, 55, 63]

})



from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=3)

kmeans.fit(df1)
import matplotlib.pyplot as plt

plt.scatter(df1['x'], df1['y'], color = "k")
labels = kmeans.predict(df1)

centroids1 = kmeans.cluster_centers_

print(centroids1)

print(labels)
df2 = pd.DataFrame({

    'x': [12, 20, 28, 18, 92, 81, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 11, 90, 81 ,31, 56, 85, 71, 13, 10, 9],

    'y': [39, 36, 30, 52, 54, 46, 88, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 99, 30, 41, 68, 74, 82 ,91, 3, 55, 63]

})

df2['closest'] = labels + 1

df2['color'] = df2['closest'].map(lambda x: colmap[x])

df2.head()
fig1 = plt.figure(figsize=(5, 5))



colors1 = ['r*', 'b*', 'g*']

plt.scatter(df2['x'], df2['y'], color=df2['color'], alpha=0.5, edgecolor='k')

for k in range(3):

    plt.plot(centroids1[k][0], centroids1[k][1], colors1[k])

plt.xlim(0, 100)

plt.ylim(0, 100)

plt.show()

print(colors)