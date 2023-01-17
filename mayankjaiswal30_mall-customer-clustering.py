import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values
from sklearn.cluster import KMeans 
wcss = [] 
for i in range(1,15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.title('The Elbow Method')
plt.xlabel('No. of cluster')
plt.ylabel('wcss')
plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], color='Red', s=100, label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], color='Blue',s=100, label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], color='Green',s = 100, label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], color='purple',s = 100, label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], color='cyan', s = 100, label = 'Cluster 5')
plt.title('Mall customer segmentation')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
