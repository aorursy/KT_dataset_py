import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from nltk.cluster.kmeans import KMeansClusterer
from nltk.cluster.em import EMClusterer
from nltk.cluster.util import cosine_distance
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Step 1: Read data into wine_data
wine_data = pd.read_csv("../input/chateau/wine_data.csv")

# Step 2: Display first five rows of data
wine_data.head()
wine_data.shape
# Create scatter plot with Pinot Noir on the x-axis and Champage on the y-axis
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'])

# Add a title to the plot
plt.title("Exhibit 2A: The Data")

# Add labels for the x and y axes
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")
kmeans2 = KMeans(n_clusters=2, init=np.array([[8, 17],[27, 1]]), n_init=1)
X_wine = wine_data[['Pinot Noir', 'Champagne']].to_numpy()
X_wine[:5]
kmeans2.fit(X_wine)
kmeans2.cluster_centers_
# Plot chart title and label the x and y axes
plt.title("Exhibit 2D: Final Centroid Locations")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=kmeans2.labels_, cmap='bwr')

# Plot the final cluster centroids in black
plt.scatter(kmeans2.cluster_centers_[:,0] ,kmeans2.cluster_centers_[:,1], marker="X", 
            color='black', s=200)
kmeans3 = KMeans(n_clusters=3, random_state=162012)
kmeans3.fit(X_wine)
kmeans3.cluster_centers_
# Plot chart title and label the x and y axes
plt.title("Exhibit 3B: Optimized clusters, k=3")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=kmeans3.labels_, cmap='brg')

# Plot the final cluster centroids in black
plt.scatter(kmeans3.cluster_centers_[:,0] ,kmeans3.cluster_centers_[:,1], marker="X", 
            color='black', s=200)
k = 2 # <-- replace "2" with your desired number of clusters

####### Step 4a: Initialize clustering #########################################
kmeansK = KMeans(n_clusters=k, random_state=162012)


####### Step 4b: Prepare input data ############################################
# Nothing to do here, as X_wine was already created earlier


####### Step 4c: Perform k-means clustering ####################################
kmeansK.fit(X_wine)


####### Step 4d: View output ###################################################
# Plot chart title and label the x and y axes
plt.title("Optimized clusters, k="+str(k))
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=kmeansK.labels_, cmap='brg')

# Plot the final cluster centroids in black
plt.scatter(kmeansK.cluster_centers_[:,0] ,kmeansK.cluster_centers_[:,1], marker="X", 
            color='black', s=200)
labels2 = kmeans2.fit_predict(X_wine)
silhouette_score(X_wine, labels2)
labels2
wine_data["Cluster (k=2)"] = labels2 + 1
wine_data.head()
labels3 = kmeans3.fit_predict(X_wine)
silhouette_score(X_wine, labels3)
wine_data["Cluster (k=3)"] = labels3 + 1
wine_data.head()
wine_data["Silhouette Value (k=2)"] = silhouette_samples(X_wine, labels2)
wine_data.head()
wine_data["Silhouette Value (k=2)"].mean()
wine_data["Silhouette Value (k=3)"] = silhouette_samples(X_wine, labels3)
wine_data.head()
wine_data.sort_values(by=["Cluster (k=2)", "Cluster (k=3)", "Silhouette Value (k=2)"],
                         ascending=[True, True, False])
silhouette_scores = [] # Initialize empty list to store silhouette scores

# Loop over different values of k and calculate the average silhouette score
for i in range(2, 50):
    
    # Initialize clustering with k = i
    kmeansI = KMeans(n_clusters=i, random_state=817910)
    
    # Apply clustering
    kmeansI.fit(X_wine)
    
    # Calculate the silhouette score when k = i
    labelsI = kmeansI.fit_predict(X_wine)
    scoreI = silhouette_score(X_wine, labelsI)
    
    # Add silhouette score at k = i to silhouette_scores
    silhouette_scores.append((i, scoreI))
    
# Plot the average silhouette score for each value of k
x,y = zip(*silhouette_scores)
plt.plot(x, y)

# Plot a vertical line at whichever k maximizes the silhouette score
maxK = x[y.index(max(y))]
plt.axvline(x=maxK, color="black")
plt.text(18, 0.4, 'k='+str(maxK), color='black')

# Label the chart and the x and y axes
plt.title("Silhouette Plot")
plt.xlabel("Number of Clusters (k)")
plt.xlim([2,50])
plt.ylabel("Average Silhouette Score")
gmm2 = GaussianMixture(n_components=2, n_init=1, random_state=994561)
gmm2.fit(X_wine)
# Plot chart title and label the x and y axes
plt.title("Gaussian Mixture Model, k=2")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
labels = gmm2.predict(X_wine)
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=labels, cmap='cool')
probAssignments = gmm2.predict_proba(X_wine)
probAssignments[:5].round(3)
labelsProb = [i[0] for i in probAssignments]
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=labelsProb, cmap='cool')
# Perform clustering
gmm3 = GaussianMixture(n_components=3, n_init=1, random_state=605973)
gmm3.fit(X_wine)

# Plot chart title and label the x and y axes
plt.title("Exhibit 5B: Gaussian Mixture Model, k=3")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
labels = gmm3.predict(X_wine)
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=labels, cmap='brg')
kmeans2Cos = KMeansClusterer(2, distance=cosine_distance, avoid_empty_clusters=True, repeats=500)
kmeans2CosClusters = kmeans2Cos.cluster(X_wine, assign_clusters=True)
# Plot chart title and label the x and y axes
plt.title("Exhibit 6B: Final Centroid Positions, k=2")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=kmeans2CosClusters, cmap='bwr')
# Perform clustering
kmeans3Cos = KMeansClusterer(3, distance=cosine_distance, avoid_empty_clusters=True, repeats=500)
kmeans3CosClusters = kmeans3Cos.cluster(X_wine, assign_clusters=True)

# Plot chart title and label the x and y axes
plt.title("Exhibit 7A: K-means Cluster Assignments")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=kmeans3CosClusters, cmap='brg')
wine_data_norm = preprocessing.normalize(wine_data[['Pinot Noir', 'Champagne']])
wine_data_norm[:5]
# Perform clustering
gmm3cos = GaussianMixture(n_components=3, random_state=144038)
gmm3cos.fit(wine_data_norm)

# Plot chart title and label the x and y axes
plt.title("Exhibit 5B: Gaussian Mixture Model, k=3")
plt.xlabel("Pinot Noir")
plt.ylabel("Champagne")

# Plot the observations color-coded by cluster
labels = gmm3cos.predict(wine_data_norm)
plt.scatter(wine_data['Pinot Noir'], wine_data['Champagne'], c=labels, cmap='brg')