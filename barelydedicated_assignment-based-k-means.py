# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from scipy.spatial import distance
import math
import seaborn as sns
import matplotlib as mpl
%matplotlib inline
sns.set_style("darkgrid")
mpl.rcParams['figure.figsize'] = (6,4)
mpl.rcParams['figure.dpi'] = 200
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
c2 = pd.read_csv("../input/C2.csv", names=['x0', 'x1'])
plt.scatter(c2['x0'],c2['x1'])
data_set = []
for index, row in c2.iterrows():
        data_set.append([row['x0'], row['x1']]) 
data_set = np.array(data_set)
        # prelim work to get every point into a list of coordinates
def init_kmeans_plus_plus(points, K, c_init = None):
    assert K>=2, "So you want to make 1 cluster?"
    # get the first centroid
    if c_init is None:
        centroids = [points[0]]
    # choice next
    for k in range (0, K-1):
        
        prob_distribution = []
        for point in points:
            proba = distance.euclidean(point,centroids[k-1])**2 # compare to last center point
            prob_distribution.append(proba)
            #normalize these values at the end so that we can now use the array as a probability distribution function
        prob_distribution = np.array(prob_distribution)/np.sum(prob_distribution)  
        #Now we will get one of these values and add it to our clusters
        centroids.append(points[np.random.choice(range(points.shape[0]), p=prob_distribution)])      
    return np.array(centroids)
cluster_points = init_kmeans_plus_plus(data_set, 3)
plt.scatter(c2['x0'],c2['x1'])
for index, point in enumerate(cluster_points):
    if index is not 0: # these points are generated
        plt.scatter(point[0],point[1], marker='*', c='red', s=50)
    if index is 0: # this is our ifrst point, which was picked staticly
        plt.scatter(point[0],point[1], marker='*', c='orange', s=50)
trials = 1000
cost = np.zeros(trials)
for _ in range(trials):
    cluster_points = init_kmeans_plus_plus(data_set, 3)
    cluster_distance = np.full(len(data_set), np.inf)
    for point_idx, point in enumerate(data_set):
        for cluster_idx, cluster_point in enumerate(cluster_points):
            if cluster_distance[point_idx] is math.inf:
                cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
                continue
            if distance.euclidean(point,cluster_point) < cluster_distance[point_idx]:
                cluster_distance[point_idx] = distance.euclidean(point,cluster_point)
    cost[_] = (math.sqrt(np.sum(cluster_distance**2, axis=0) /len(data_set)))
#print('3-means costs for  trials= ', trials,' is cost= ', cost)
y = []
i=0
totalFrac = 1/trials
cost = sorted(cost)
for z in range(len(cost)):
    y.append(cost.index(cost[z])*totalFrac)
    if math.fabs(cost[z] - 3.699) < 0.1:
        i = i +1
print('Close 3-means compared to Gonzalez:',i)
plt.title("cdf of 3-means cost associated with k-means++ algorithm")
plt.xlabel("cost")
plt.ylabel("fraction of success")
plt.xlim([np.min(cost),np.max(cost)])
plt.ylim([0.00,1.05])
plt.plot(np.array(cost), np.array(y))
plt.show()