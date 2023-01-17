# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans

import numpy as np



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Initialisation des points centraux

# random ou manuellement

# On considére que le cluster correspond à l'indice dans le tableau centers

K = 4

centers = [[20,15], [50,35], [10,9], [34,60]]

df = pd.DataFrame(centers, columns=["x", "y"])

df.plot.scatter(x="x", y="y")
# Compute distance between 2 points

def euclidean_distance(p1, p2):

    # p --> (x,y)

    somme_square = 0

    for i in range(len(p1)):

        somme_square += (p1[i]-p2[i])**2

    return np.sqrt(somme_square)
# calcul of the mean with points as inputs

def get_mean(points):

    nb_points = len(points)

    size_point = len(points[0])

    mean_point = np.zeros(size_point)

    for point in points:

        for i in range(size_point):

            mean_point[i]+=point[i]

    return mean_point/nb_points
# Calcul of the variance

def get_variance(points):

    nb_points = len(points)

    mean_p = mean(points)

    print("la moyenne est de : ", mean_p)

    somme = 0

    for p in points:

        dist = euclidean_distance(p,mean_p)**2

        print("la distance de ", p, "à ", mean_p, " est de ", dist)

        somme+= dist

    return somme/nb_points
# Normalized the data

def get_normalized_data(data):

    data_norm = []

    mean = get_mean(data)

    variance = get_variance(data)

    for x in data:

        data_norm.append((x-mean)/variance)

    return data_norm
# Find the nearest central point

def find_nearest_center_cluster(p1, centers):

    min_dist = euclidean_distance(p1, centers[0])

    nearest_center_ind = 0

    for i_c in range(len(centers)):

        d = euclidean_distance(p1, centers[i_c])

        #print(p1, " --> ", centers[i_c])

        #print("\tdistance : ", d)

        if d < min_dist:

            nearest_center_ind = i_c

            min_dist = d

    return nearest_center_ind
def apply(vect, func, res=None):

    if len(vect)==0:

        return res

    else:

        return apply(vect[:-1], func, func(res,vect[-1]))  
arr = [(1,2),(1,2),(1,2)]



def plus(a,b):

    return a+b



def mult(a,b):

    return a*b



# Trouve le centre d'un nuage de points de N dimension

def find_center_point(points):

    nb_points = len(points)

    size_point = len(points[0])

    center_coord = []

    for i in range(size_point):

        res = apply([p[i] for p in points],plus,0)

        center_coord.append(res/nb_points)

    return center_coord
# Iteration :

def iterate_clustering(data_points, centers):

    K = len(centers)

    clusters = []

    for k in range(K):

        clusters.append([])

        

    for point in data_points:

        cluster = find_nearest_center_cluster(point, centers)

        clusters[cluster].append(point)

        print("the point with point",point, "is among the cluster n°",cluster)

    return clusters
# Test iterate clustering

data_points = [[54,23],[26,43],[18,32],[34,25],[42,31],[26,29],[54,62]]

centers = [[20,15], [50,35], [10,9], [34,60]]



clusters = iterate_clustering(data_points, centers)
# Recalculate the centers

def recalculate_centers(new_clusters, old_centers):

    new_centers = []

    for i in range(len(new_clusters)):

        if new_clusters[i]==[]:

            new_centers.append(old_centers[i])

        else:

            new_centers.append(find_center_point(new_clusters[i]))

    return new_centers



new_centers = recalculate_centers(clusters, centers)
# Verify if center are stable or not by verify if center changed or not

def centers_changed(old_centers, new_centers):

    for ocenter in old_centers:

        if ocenter not in new_centers:

            return 1

        else:

            return 0
import matplotlib as plt

# Testing the iteration function

data_points = np.array([[54,23],[26,43],[18,32],[34,25],[42,31],[26,29],[54,62]])

clusters = iterate_clustering(data_points, centers)

new_centers = recalculate_centers(clusters, centers)

while(centers_changed(centers,new_centers)):

    clusters = iterate_clustering(data_points, new_centers)

    n_centers = recalculate_centers(clusters, centers)

    centers = new_centers

    new_centers = n_centers

print(clusters)

print(new_centers)

# For c in clusters:

#    x,y = [data_points[p] for p in clusters]
# Compare with kmeans from scikit learn

kmeans_model = KMeans(n_clusters=4, init=np.array(centers))

print(kmeans_model)
# Model fit the data

kmeans_model.fit(data_points)

print(kmeans_model.cluster_centers_)