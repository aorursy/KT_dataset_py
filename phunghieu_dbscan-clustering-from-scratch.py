import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
DATA_PATH = '../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv'



EPSI = 10

MIN_POINTS = 3
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
class Point(object):

    def __init__(self, coordinate):

        self.coordinate = coordinate

        self.cluster_idx = None

        

    def is_clustered(self):

        return self.cluster_idx is not None

    

    def cluster(self, cluster_idx):

        self.cluster_idx = cluster_idx





class DBSCAN(object):

    def __init__(self, epsi, min_points):

        self.epsi = epsi

        self.min_points = min_points

    

    @staticmethod

    def _compute_distance(x1, x2):

        return np.sum((x1 - x2)**2)**0.5



    def _find_neighbor_indices(self, core_point, other_points):

        neighbor_indices = []

        

        for idx in range(len(other_points)):

            if self._compute_distance(core_point.coordinate, other_points[idx].coordinate) <= self.epsi:

                neighbor_indices.append(idx)

                

        return neighbor_indices



    def fit(self, df, feature_cols=['Annual Income (k$)', 'Spending Score (1-100)']):

        points = []

        for idx, row in df.loc[:, feature_cols].iterrows():

            points.append(Point(row.to_numpy()))

                    

        free_point_indices = set(range(len(points)))



        core_point_indices = set()

        cluster_count = 0



        while len(free_point_indices) > 0:

            if len(core_point_indices) == 0:

                start_idx = random.choice(list(free_point_indices))

                core_point_indices.add(start_idx)

                free_point_indices.remove(start_idx)

                points[start_idx].cluster(cluster_count)



            while len(core_point_indices) > 0:

                picked_core_point_idx = core_point_indices.pop()

                

                neighbor_indices = self._find_neighbor_indices(points[picked_core_point_idx], points)

                

                if len(neighbor_indices) >= (self.min_points - 1):

                    for neighbor_idx in neighbor_indices:

                        if neighbor_idx in free_point_indices:

                            points[neighbor_idx].cluster(cluster_count)

                            if neighbor_idx in free_point_indices:

                                core_point_indices.add(neighbor_idx)

                                free_point_indices.remove(neighbor_idx)

                    

            cluster_count += 1

                    

        return points
model = DBSCAN(epsi=EPSI, min_points=MIN_POINTS)
points = model.fit(df)
x = []

y = []

cluster = []



for point in points:

    _x, _y = point.coordinate

    x.append(_x)

    y.append(_y)

    cluster.append(point.cluster_idx)
new_df = pd.DataFrame({

    'Annual Income (k$)': x,

    'Spending Score (1-100)': y,

    'cluster': cluster

})
new_df
fig, ax = plt.subplots(figsize=(16, 8))



sns.scatterplot(data=new_df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', palette='pastel', ax=ax)

plt.show()