# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#load dataset

customers = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
#first few entries of dataset

customers.head()
#dataset info

customers.info()
plt.figure()

plt.hist(customers['Gender'])

plt.title('Distribution of Gender')

plt.show()
import plotly.io as pio

pio.renderers.default='notebook'
import plotly.express as px



fig = px.scatter_3d(customers, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',

              color='Gender')

fig.show()
#check unique values in Gender

print(customers['Gender'].unique())
#replace  Male/Female with 0/1

customers['Gender'].replace({'Male':0, 'Female':1}, inplace=True)

#check

print(customers['Gender'].unique())
#descriptive statistics

customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe()
#linear correlation coefficients between Age, Annual Income, and Spending Score

corr_customers = customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()

mask = np.triu(np.ones_like(corr_customers, dtype=bool))

sns.heatmap(corr_customers, mask=mask,annot=True, cmap='BuPu')

plt.show()
import plotly.io as pio

pio.renderers.default='notebook'



import plotly.graph_objs as go



data = go.Scatter3d(x = customers['Age'], y = customers['Annual Income (k$)'], z = customers['Spending Score (1-100)'], 

                    mode ='markers', marker = dict(size = 4, color = 'crimson',  line=dict(width=2, color='DarkSlateGrey')))



layout =  dict(title = 'Customers',

              scene = dict(xaxis= dict(title= 'Age',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Annual Income',ticklen= 5,zeroline= False),

              zaxis= dict(title= 'Spending Score',ticklen= 5,zeroline= False))

             )



fig = go.Figure(dict(data = data, layout = layout))



fig.show()
#scaling the features

from sklearn.preprocessing import minmax_scale



for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:

    customers[col + '_scaled'] = minmax_scale(customers[col])

    

#check

customers.columns
def findClosestCentroids(X, centroids):

    

    '''

    Calculates the closest centroid in centroids for each training example in X.

    Returns vector of centroid assignments for each training example.

    '''

    

    #set K

    K = centroids.shape[0]

    

    #vector of cluster assignments

    idx = np.zeros(X.shape[0])

    

    dist = np.zeros(K)

    for i in range(X.shape[0]):

        for k in range(K):

            dist[k] = np.sum((X[i,:] - centroids[k,:])**2)**0.5

        idx[i] = np.argmin(dist)

        

    return idx
def computeCentroids(X, idx, K):

    

    '''

    Calculates new centroids by computing the mean of the 

    data points assigned to each centroid. Returns matrix

    where each row is a new centroid's point.

    '''

    

    #no. of data points

    m = X.shape[0]

    #dimension of points

    n = X.shape[1]

    

    centroids = np.zeros((K,n))

    

    for k in range(K):

        count = 0

        s = np.zeros((1,n))

        for i in range(m):

            if idx[i] == k:

                s = s + X[i,:]

                count += 1

        centroids[k,:] = s/count

        

    return centroids
def RandInitialCentroids(X, K):

    

    '''

    Initializes K centroids by randomly selecting K points in X.

    '''

    

    centroids = np.zeros((K, X.shape[1]))

    

    #Randomly reorder the indicies of examples

    randidx = np.random.permutation(range(X.shape[0]))

    #Take the first K examples

    centroids = X[randidx[0:K],:]

    

    return centroids
def kMeansDistortion(X, idx, centroids):

    

    '''

    Calculates the average distance between the examples and the 

    centroid of the cluster to which each example has been assigned.

    '''

    

    #no. of data points

    m = X.shape[0]

    

    distortion = 0

    

    for i in range(X.shape[0]):

        closest = int(idx[i])

        distance = np.sum((X[i,:] - centroids[closest])**2)

        distortion = distortion + distance

        

    distortion = distortion/m

    

    return distortion
def kMeans(X, K, max_iters):

           

    '''

    Run the kmeans algorithm for specified number of iterations 

    and returns final centroids, index of closest centroids for 

    each example (idx), final distortion, and distortion history.

    '''

    distortion_history = []

    distortion = 0

    centroids = RandInitialCentroids(X, K)       



    for i in range(max_iters):

        idx = findClosestCentroids(X, centroids)

        distortion = kMeansDistortion(X, idx, centroids)

        distortion_history.append(distortion)

        centroids = computeCentroids(X, idx, K)

        

    return centroids, idx, distortion, distortion_history           

           
centroids, idx, distortion, distortion_history = kMeans(np.array(customers[['Age_scaled', 'Annual Income (k$)_scaled', 'Spending Score (1-100)_scaled']]), 5, 30)
plt.figure()

plt.plot(distortion_history)

plt.title('Distortion History')

plt.xlabel('Iteration')

plt.ylabel('Distortion')

plt.show()
print(distortion_history)
#run kmeans with specified different random initialisations and pick one with lowest distortion

def kMeansRuns(X, K, max_iters, init_runs):

    '''

    Run the kMeans algorithm for specified number of random initialisations, init_runs, 

    and return result with lowets distortion.

    '''

    for r in range(init_runs):

        if r == 0:

            centroids, index, distortion, distortion_hist = kMeans(X, K, max_iters)

            distortion_lowest = distortion

        else:

            current_centroids, current_index, distortion, current_distortion_hist = kMeans(X, K, max_iters)

            if distortion_lowest > distortion:

                centroids= current_centroids

                index = current_index

                distortion_lowest = distortion

                

    return centroids, index, distortion_lowest

from scipy.spatial import distance



def SilhouetteScore(x_i, X, idx, K):

    '''

    For given training example (with index x_i), calculates  and returns its silhouette score.

    

    First, the function calculates the average distance, a_i, between the given training example

    and all other points in the cluster it belongs to. Then, the function calculates the average distance

    between the training example and all other points not in its own cluster, and picks the cluster with 

    the smallest average distance. Using a_i and b_i, it calculates the silhouette score of the given 

    training example.

    '''

    #calculate average distance between x_i and all points in its cluster

    

    #training example 

    point = X[x_i]

    #cluster index of training example

    idx_point = idx[x_i]

    

    #list of distances between point and other points in own cluster

    own_cluster_distances = np.empty(0)

    #loop over training examples' assigned cluster index, find points in 

    #own cluster and calculate euclidean distance

    for i in range(idx.shape[0]):

        if idx[i] == idx_point:

            own_cluster_distances = np.append(own_cluster_distances, distance.euclidean(point, X[i]))

    

    #average distance between point and all other points in own cluster

    a_i = np.sum(own_cluster_distances)/(own_cluster_distances.shape[0])

    

    #for each k in range K, calculate average distance between point and all other points in cluster k

    avg_cluster_distances = np.empty(0)

    #range of K without given trainig example's own cluster

    other_clusters = [r for r in range(K) if r != idx_point]

    

    for k in other_clusters:

        #distances between point and all points in cluster k

        k_distances = np.empty(0)

        #all points in cluster k

        k_cluster = X[idx==k]

        #number of points in cluster k

        k_len = k_cluster.shape[0]

        for n in range(k_len):

            k_distances = np.append(k_distances, distance.euclidean(point, k_cluster[n]))

        #average distance between point and all points in cluster k appended to

        #avg_cluster_distances array

        if k_len != 0:

            avg_cluster_distances = np.append(avg_cluster_distances, np.sum(k_distances)/k_len)

        else:

            avg_cluster_distances = np.append(avg_cluster_distances, 0)

        

        

    #find closest cluster in avg_cluster_distances

    b_i = np.min(avg_cluster_distances)

    

    silhouette_score = (b_i - a_i)/np.max([a_i, b_i])

    

    

    return silhouette_score 
def AverageSilhouette(X, idx,  K):

    '''

    Calculates and returns the average silhoutte for given number of clusters, K.

    

    Average silhouette is the average of the silhouette scores of all the training examples.

    '''

    silhouette_scores = np.empty(0)

    #loop over all training examples and calculate their silhouette score

    for i in range(X.shape[0]):

        silhouette_i = SilhouetteScore(i, X, idx, K)

        silhouette_scores = np.append(silhouette_scores, silhouette_i)

    

    #calculate average of all scores

    avg_silhouette = np.sum(silhouette_scores)/len(silhouette_scores)

    

    return avg_silhouette

        

def PlotAvgSilhouettes(X, K_range, max_iters, init_runs):

    

    '''

    Runs kMeans and plots the average silhouette scores for 

    each number of clusters in K_range.

    

    '''

    clusters_avg_sil = np.empty(0)

    #minimum of K_range must be 2

    for K in range(2, K_range+1):

        centroids, idx, distortion_lowest = kMeansRuns(X, K, max_iters, init_runs)

        k_avg_sil = AverageSilhouette(X, idx,  K)

        clusters_avg_sil = np.append(clusters_avg_sil, k_avg_sil)

        

    

    plt.figure(figsize = (12.8, 9.6))

    plt.plot(np.arange(2,K_range+1,1), clusters_avg_sil)

    plt.title('Average Silhouette for number of clusters K')

    plt.xlabel('K')

    plt.ylabel('Average Silhouette')

    plt.show()

    

    

    return None
#features as an array, X

X = np.array(customers[['Age_scaled', 'Annual Income (k$)_scaled', 'Spending Score (1-100)_scaled']])
import warnings

warnings.simplefilter('error')



#calculate and plot graph of average silhouettes for 2-15 clusters

PlotAvgSilhouettes(X, 15, 20, 50)
centroids, ind, distortion = kMeansRuns(X, 9, 20, 50)
#extract unscaled features into variable C so we can plot and understand the results

C = np.array(customers[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])



#get clusters

cluster0 = C[ind==0]

cluster1 = C[ind==1]

cluster2 = C[ind==2]

cluster3 = C[ind==3]

cluster4 = C[ind==4]

cluster5 = C[ind==5]

cluster6 = C[ind==6]

cluster7 = C[ind==7]

cluster8 = C[ind==8]
print(cluster0.shape)

print(cluster1.shape)

print(cluster2.shape)

print(cluster3.shape)

print(cluster4.shape)

print(cluster5.shape)

print(cluster6.shape)

print(cluster7.shape)

print(cluster8.shape)
#cluster 3d scatter plot



import plotly.graph_objs as go





trace0 = go.Scatter3d(x = cluster0[:,0], y = cluster0[:,1], z = cluster0[:,2], 

                      mode = 'markers', name='Cluster0', marker = dict(size = 4, color = 'black'))



trace1 = go.Scatter3d(x = cluster1[:,0], y = cluster1[:,1], z = cluster1[:,2], 

                      mode = 'markers', name='Cluster1', marker = dict(size = 4, color = 'green'))



trace2 = go.Scatter3d(x = cluster2[:,0], y = cluster2[:,1], z = cluster2[:,2], 

                      mode = 'markers', name='Cluster2', marker = dict(size = 4, color =  'chartreuse'))



trace3 = go.Scatter3d(x = cluster3[:,0], y = cluster3[:,1], z = cluster3[:,2], 

                      mode = 'markers', name='Cluster3', marker = dict(size = 4, color =  'maroon'))



trace4 = go.Scatter3d(x = cluster4[:,0], y = cluster4[:,1], z = cluster4[:,2], 

                      mode = 'markers', name='Cluster4', marker = dict(size = 4, color =  'hotpink'))



trace5 = go.Scatter3d(x = cluster5[:,0], y = cluster5[:,1], z = cluster5[:,2], 

                      mode = 'markers', name='Cluster5', marker = dict(size = 4, color =  'crimson'))



trace6 = go.Scatter3d(x = cluster6[:,0], y = cluster6[:,1], z = cluster6[:,2], 

                      mode = 'markers', name='Cluster6', marker = dict(size = 4, color =  'cyan'))





trace7 = go.Scatter3d(x = cluster7[:,0], y = cluster7[:,1], z = cluster7[:,2], 

                      mode = 'markers', name='Cluster7', marker = dict(size = 4, color =  'darkblue'))





trace8 = go.Scatter3d(x = cluster8[:,0], y = cluster8[:,1], z = cluster8[:,2], 

                      mode = 'markers', name='Cluster8', marker = dict(size = 4, color =  'chocolate'))



data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]





layout = dict(title = 'Mall Segmentation Clusters',

              scene = dict(xaxis= dict(title= 'Age',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Annual Income',ticklen= 5,zeroline= False),

              zaxis= dict(title= 'Spending Score',ticklen= 5,zeroline= False))

             )



fig = go.Figure(dict(data = data, layout = layout))



fig.show()

#average silhouette score

print('The average silhouette score of this model with 9 clusters is {:0.2f}.'.format(AverageSilhouette(X, ind,  9)))