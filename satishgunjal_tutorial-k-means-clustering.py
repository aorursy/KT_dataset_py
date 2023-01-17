import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
#df = pd.read_csv('https://raw.githubusercontent.com/satishgunjal/datasets/master/Mall_Customers.csv')

df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

print("Shape of the data= ", df.shape)

df.head()
plt.figure(figsize=(10,6))

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.title('Unlabelled Mall Customer Data')
# Since we are going to use Annual Income and Spending Score  columns only, lets create 2D array of these columns for further use

X = df.iloc[:, [3,4]].values

X[:5] # Show first 5 records only
clustering_score = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)

    kmeans.fit(X)

    clustering_score.append(kmeans.inertia_) # inertia_ = Sum of squared distances of samples to their closest cluster center.

    



plt.figure(figsize=(10,6))

plt.plot(range(1, 11), clustering_score)

plt.scatter(5,clustering_score[4], s = 200, c = 'red', marker='*')

plt.title('The Elbow Method')

plt.xlabel('No. of Clusters')

plt.ylabel('Clustering Score')

plt.show()
kmeans= KMeans(n_clusters = 5, random_state = 42)



# Compute k-means clustering

kmeans.fit(X)



# Compute cluster centers and predict cluster index for each sample.

pred = kmeans.predict(X)



pred
df['Cluster'] = pd.DataFrame(pred, columns=['cluster'] )

print('Number of data points in each cluster= \n', df['Cluster'].value_counts())

df
plt.figure(figsize=(10,6))

plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown', label = 'Cluster 0')

plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green', label = 'Cluster 1')

plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue', label = 'Cluster 2')

plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple', label = 'Cluster 3')

plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange', label = 'Cluster 4')



plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 300, c = 'red', label = 'Centroid', marker='*')



plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.title('Customer Clusters')
def plot_k_means_progress(centroid_history,n_clusters, centroid_sets, cluster_color):

    """

    This function will plot the path taken by the centroids

    

    I/P:

    * centroid_history: 2D array of centroids. Each element represent the centroid coordinate. 

      If there are 5 clusters then first set contains initial cluster cordinates

      (i.e. first 5 elements) and then k_means loop will keep appending new cluster coordinates for each iteration

    * n_clusters: Total number of clusters to find

    * centroid_sets: At the start we set random values as our first centroid set. K-Means loop will keep adding 

    new centroid sets to centroid_history. Since we are ploting the path of centroid locations, centroid set value 

    will be K-Means loop iteration number plus 1 for initial centroid set. 

    So its value will be from 2 to K-Means loops max iter plus 1

    * cluster_color: Just to have same line and cluster color

    

    O/P: Plot the centroid path

    """

    c_x = [] # To store centroid X coordinated

    c_y=[]   # To store the centroid Y coordinates

    for i in range(0, n_clusters):

        cluster_index = 0

        for j in range(0, centroid_sets):

            c_x = np.append(c_x, centroid_history[:,0][i + cluster_index])

            c_y = np.append(c_y, centroid_history[:,1][i + cluster_index])

            cluster_index = cluster_index + n_clusters

            # if there are 5 clusters then first set contains initial cluster cordinates and then k_means loop will keep appending new cluster coordinates for each iteration

        

        plt.plot(c_x, c_y, c= cluster_color['c_' + str(i)], linestyle='--')

        

        # Reset coordinate arrays to avoid continuous lines

        c_x = []

        c_y=[]
plt.figure(figsize=(10,6))



# Random Initialization of Centroids

plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'])

initial_centroid = np.array([[10, 2], [50,100], [130,20], [50,15], [140,100]])



plt.scatter(initial_centroid[:,0], initial_centroid[:, 1],s = 200, c = 'red', label = 'Random Centroid', marker='*')

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.title('Random Initialization of Centroids')



# K-Means loop of assignment and move centroid steps

centroid_history = []

centroid_history = initial_centroid

#

cluster_color= {'c_0':'brown','c_1':'green','c_2':'blue','c_3':'purple','c_4':'orange'}

n_clusters = 5

for i in range(1,6):

    kmeans= KMeans(n_clusters, init= initial_centroid, n_init= 1, max_iter= i, random_state = 42)  #n_init= 1 since our init parameter is array

    

    # Compute cluster centers and predict cluster index for each sample

    pred = kmeans.fit_predict(X)



    plt.figure(figsize=(10,6))

    plt.scatter(X[pred == 0, 0], X[pred == 0, 1], c = 'brown', label = 'Cluster 0')

    plt.scatter(X[pred == 1, 0], X[pred == 1, 1], c = 'green', label = 'Cluster 1')

    plt.scatter(X[pred == 2, 0], X[pred == 2, 1], c = 'blue', label = 'Cluster 2')

    plt.scatter(X[pred == 3, 0], X[pred == 3, 1], c = 'purple', label = 'Cluster 3')

    plt.scatter(X[pred == 4, 0], X[pred == 4, 1], c = 'orange', label = 'Cluster 4') 

    

    plt.scatter(centroid_history[:,0], centroid_history[:, 1],s = 50, c = 'gray', label = 'Last Centroid', marker='x')

    

    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],s = 200, c = 'red', label = 'Centroid', marker='*')

    

    centroid_history = np.append(centroid_history, kmeans.cluster_centers_, axis=0)

    

    plt.xlabel('Annual Income')

    plt.ylabel('Spending Score')

    plt.legend()

    plt.title('Iteration:' + str(i) + ' Assignment and Move Centroid Step')

    

    centroid_sets = i + 1 # Adding one for initial set of centroids

    plot_k_means_progress(centroid_history,n_clusters, centroid_sets, cluster_color)