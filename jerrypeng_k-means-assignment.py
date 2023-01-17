# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.DataFrame({

    'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],

    'y': [1, 2, 4, 7, 5, 5, 4.5]

})
k = 2
centroids = {

    1: [1.8, 2.3],

    2: [4.1, 5.4]

}
fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color = 'k')

colmap = {1: 'r', 2: 'g'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 8)

plt.ylim(0, 8)

plt.show()
# xi and xj should be np.array

def distance(xy, mu, dis='Euclidean'):

    if dis == 'Cityblock':

        return np.abs( np.sum(xy.sub(mu), axis = 1))

    else:

        return np.sqrt( np.sum((xy.sub(mu) ) ** 2, axis = 1) )
def assignment(df, centroids):

    # assignment

    for i in centroids.keys():

        df['distomu_{}'.format(i)] = (

            distance(df[['x', 'y']], centroids[i])

        )

    centroid_distance_cols = ['distomu_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distomu_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df
df = assignment(df, centroids)

print(df.head())
fig = plt.figure(figsize=(5, 5))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 8)

plt.ylim(0, 8)

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

plt.xlim(0, 8)

plt.ylim(0, 8)

for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=0.2, head_length=0.3, fc=colmap[i], ec=colmap[i])

plt.show()
import threading, time

import numpy as np  # linear algebra

import pandas as pd

import copy

import matplotlib.pyplot as plt

# matplotlib inline



df = pd.DataFrame({

    'x': [1, 1.5, 3, 5, 3.5, 4.5, 3.5],

    'y': [1, 2, 4, 7, 5, 5, 4.5]

})



# centroids = {

#     1: [1.8, 2.3],

#     2: [4.1, 5.4]

# }



centroids = {

    1: [1.8, 2.3],

    2: [4.1, 5.4]

}



# xi and xj should be np.array

def distance(xy, mu, dis='Euclidean'):

    if dis == 'Cityblock':

        return np.abs(np.sum(xy.sub(mu), axis=1))

    else:

        return np.sqrt(np.sum((xy.sub(mu)) ** 2, axis=1))





def assignment(df, centroids):

    # assignment

    colmap = {1: 'r', 2: 'g'}

    for i in centroids.keys():

        df['distomu_{}'.format(i)] = (

            distance(df[['x', 'y']], centroids[i], 'Cityblock')

        )

    centroid_distance_cols = ['distomu_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distomu_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df





def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])

        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])

    return k



def getDataAndUpdate(df, centroids):

    colmap = {1: 'r', 2: 'g'}

    while True:

        iter = 2;



        closest_centroids = df['closest'].copy(deep=True)

        centroids = update(centroids)

        df = assignment(df, centroids)

        # draw scatter

        plt.clf()

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

        for i in centroids.keys():

            plt.scatter(*centroids[i], color=colmap[i])

        plt.xlim(0, 8)

        plt.ylim(0, 8)

        plt.title("Iteration: %d" %iter)

        iter = iter + 1

        plt.draw()

        plt.pause(1)



        if closest_centroids.equals(df['closest']):

            break





fig = plt.figure(figsize=(5, 5))

df = assignment(df, centroids)

plt.clf()

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

colmap = {1: 'r', 2: 'g'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 8)

plt.ylim(0, 8)

plt.title("Iteration: 1")

plt.draw()

plt.ion()

plt.pause(2)



getDataAndUpdate(df, centroids)