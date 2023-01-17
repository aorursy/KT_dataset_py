import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
%%javascript
IPython.OutputArea.auto_scroll_threshold = 9999;
# Random sample generation
X, y = make_blobs(n_samples=5000, centers=[[70,70], [-30, -20], [10, 50], [40, -10]], cluster_std=15)
# putting the samples in the data frame with x1 and x2 as the features
df = pd.DataFrame({'x1':X[:,0],'x2':X[:,1]})
df['x1'] = df['x1'].apply(lambda x: int(x*10)) 
df['x2'] = df['x2'].apply(lambda x: int(x*10)) 
plt.scatter(df['x1'], df['x2'], alpha=0.3, edgecolor='k')
# Initial Custer centroid
cluster = {}

k=4 # cluster
for i in range(1,k+1):
    cluster['C'+str(i)] = (np.random.randint(low=df.x1.min(),high=df.x1.max()),np.random.randint(low=df.x2.min(),high=df.x2.max()))
    
plt.scatter(df['x1'], df['x2'], alpha=0.3,  edgecolor='k')
for i in cluster.keys():
    plt.scatter(cluster[i][0],cluster[i][1],color='k', marker="*" ,s=150)
cluster
#color encoding for the centroid
colmap = {1: 'r', 2: 'g', 3: 'b',4:'y'}
plt.scatter(df['x1'], df['x2'], alpha=0.3, edgecolor='k')

for i in cluster.keys():
    plt.scatter(cluster[i][0],cluster[i][1],color=colmap[int(i[-1])], marker="X" ,s=100)
# now we have the x1 and x2 variables and random variable for the centroid
def assignment_step(df,cluster):
    for i in cluster.keys():
            df['distance from {}'.format(i)] = np.sqrt((df.x1 - cluster[i][0])**2 + (df.x2 - cluster[i][1])**2)
    centroid_distance_cols = ['distance from {}'.format(i) for i in cluster.keys()]
    df['center'] = df[centroid_distance_cols].idxmin(axis=1)
    df['center'] = df['center'].map(lambda x: int(x.lstrip('distance from C')))
    df['color'] = df['center'].map(lambda x: colmap[x])
     
assignment_step(df,cluster)
plt.scatter(df['x1'], df['x2'], color=df['color'], alpha=0.3, edgecolor='k')

for i in cluster.keys():
    plt.scatter(cluster[i][0],cluster[i][1],color=colmap[int(i[-1])], marker="X" ,s=100)
import copy
old_centroid = copy.deepcopy(cluster)
old_centroid
def update_centroid(k):
    cluster = {}
    for i in k.keys():
        c = int(i[-1])
        new_point = (df[df['center']==c]['x1'].mean(), df[df['center']==c]['x2'].mean())
        cluster[i] = new_point
    return cluster

cluster_centroid = {}
for x in cluster.keys():
    cluster_centroid[x] = []
def k_means_clustering(iteration=10):
    global cluster # step 1 using the random generated cluster centroids
    global cluster_centroid
    for i in range(iteration):
        assignment_step(df,cluster) # step 2
        cluster = update_centroid(cluster)# step 1
        for c in cluster.keys():
            cluster_centroid[c].append(cluster[c])
        # visualizing the convergence of the clusters    
        plt.figure(figsize=(8,6))
        plt.scatter(df['x1'], df['x2'], color=df['color'], alpha=0.3, edgecolor='k')
        plt.title("Iteration Number {}".format(i+1))
        for x in cluster_centroid.keys():

            plt.scatter(*zip(*cluster_centroid[x]), marker="*", s=100,color="k")
            plt.plot(*zip(*cluster_centroid[x]), color='k',linewidth=2) 
        plt.show()

        
k_means_clustering()
plt.figure(figsize=(10,8))
plt.scatter(df['x1'], df['x2'], color=df['color'], alpha=0.3, edgecolor='k')
for x in cluster_centroid.keys():
    
    plt.scatter(*zip(*cluster_centroid[x]), marker="*", s=100,color="k")
    plt.plot(*zip(*cluster_centroid[x]), color='k',linewidth=2)