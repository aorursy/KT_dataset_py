import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from shutil import copyfile, rmtree



%matplotlib inline
copyfile(src = "../input/kmeansutilsmodule/KMeansUtils.py", dst = "../working/KMeansUtils.py")



# some utility functions to test and visualize K-Means

from KMeansUtils import *
def _distance(point_a, point_b):

    """Calculates euclidean distance between 2 points."""

    

    # axis=1 allows vectorized function - that is, calculating the distance between point_a and many point_b's

    return np.sqrt(np.sum(np.power(point_a - point_b, 2), axis=1)).reshape(-1, 1)
def cluster_assignment(data, centroids):

    """Assigns cluster to each of the data points."""

    

    distances = np.concatenate([_distance(cent, data) for cent in centroids], axis=1)

    return distances.argmin(axis=1)
def move_clusters(assignments, data, clusters):

    """Moves cluster centroids to the average location of it's assigned data points."""

    

    cluster_ids = np.unique(assignments)

    new_cluster_locs = np.zeros(shape=clusters.shape, dtype=float)

    

    for cluster_id in cluster_ids:

        cur_cluster_points_idx = np.where(assignments==cluster_id)

        cur_cluster_points = data[cur_cluster_points_idx]    

        new_cluster_locs[cluster_id, :] = np.mean(cur_cluster_points, axis=0)

    

    return new_cluster_locs
def distortion_function(assignments, data, clusters):

    """Calculates K-Means cost function - mean of squared distances of each sample from it's assigned cluster centroid."""

    

    cluster_ids = np.unique(assignments)

    cost = 0

    

    for cluster_id in cluster_ids:

        cur_cluster_points_idx = np.where(assignments==cluster_id)

        cur_cluster_points = data[cur_cluster_points_idx]

        

        cost += np.sum(_distance(clusters[cluster_id], cur_cluster_points))

    

    cost /= data.shape[0]

    

    return cost
def random_start(data, k):

    """Randomly chooses k points from the data to be used as cluster centroids."""

    

    return data[np.random.choice(data.shape[0], size=k, replace=False)]
def kmeans(data, k, iterations, scatter='2D', save_scatter=False, save_dir_path=''):

    """K-Means algorithm (with scatter option for 2 and 3 dimensional data)."""

    

    # initialize centroids locations

    centroids = random_start(data=data, k=k)

    

    # instantiate an array to log cost function value at each iteration

    cost = np.zeros((10+1), dtype=float)

    

    scatter_plots = []



    for i in range(iterations):

        

        # assign clusters to each data point

        assignments = cluster_assignment(data, centroids)

        

        # compute cost function value

        cost[i] = distortion_function(assignments, data, centroids)

        

        # visualize K-Means clusters and data assignments

        scatter_plots.append(

            cluster_scatter(data, centroids, assignments, i,

            scatter, save=save_scatter, save_dir_path=save_dir_path)

        )

        

        # move cluster centroids

        centroids = move_clusters(assignments, data, centroids)

        

    # compute cost function value

    cost[iterations] = distortion_function(assignments, data, centroids)

    

    #visualize K-Means clusters and data assignments

    last_scatter = cluster_scatter(

        data, centroids, assignments, iterations, scatter,

        save=save_scatter, save_dir_path=save_dir_path

    )

    scatter_plots.append(last_scatter)

    

    return centroids, assignments, cost, last_scatter, scatter_plots
# Generates random data points around given Centroids locations.

# the tuples are of form: ((x1, x2, ..., xn), variance, number of samples).

location_to_generate = [

    ((0, 0), 2, 50),

    ((15, 0), 2, 50),

    ((0, 15), 2, 50),

    ((15, 15), 2, 50),

    ((-5, 10), 3, 100)

]

data = generate_centroids_data(location_to_generate)

data.shape
scatter_2d(data)
# directory path for saving the scatterplots of each iteration, later to create an animated gif

save_dir_path = '2D_imgs'



# remove directory if already exists

if os.path.isdir(save_dir_path):

    rmtree(save_dir_path)
k = 6

iterations = 8
centroids, assignments, cost, last_scatter, scatter_plots = kmeans(

    data=data, k=k, iterations=8,

    scatter='2D', save_scatter=True, save_dir_path='2D_imgs')
plot_distortion(cost_values=cost)
create_gif(gif_name='2D', dir_path=save_dir_path)

Image(filename=f'{os.path.join(save_dir_path, "2D.gif")}')
last_scatter
location_to_generate = [

    ((0, 0, 0), 2, 50),

    ((10, 0, 0), 1, 50),

    ((0, 10, 0), 3, 50),

    ((0, 0, 10), 1, 50),

    ((10, 10, 0), 2, 50),

    ((10, 0, 10), 1, 50),

    ((0, 10, 10), 3, 50),

    ((10, 10, 10), 1, 50)

]

data = generate_centroids_data(location_to_generate)

data.shape
scatter_3d(data)
save_dir_path = '3D_imgs'



# remove directory if already exists

if os.path.isdir(save_dir_path):

    rmtree(save_dir_path)
k = 8

iterations = 10
centroids, assignments, cost, last_scatter, scatter_plots = kmeans(

    data=data, k=k, iterations=iterations,

    scatter='3D', save_scatter=True, save_dir_path='3D_imgs')
plot_distortion(cost_values=cost)
create_gif(gif_name='3D', dir_path=save_dir_path)

Image(filename=f'{os.path.join(save_dir_path, "3D.gif")}')
plotly_3d_scatter(data, centroids, assignments, iteration=10)