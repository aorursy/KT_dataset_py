#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#importing the Iris dataset with pandas

dataset = pd.read_csv('../input/iris/Iris.csv')

#Drop the ID column and select only the feature columns

#features_data = dataset.iloc[:,[1,2,3,4]].values

features_data = dataset[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

features_data
#Using KMeans method - Assuming we do not know the number of clusters of this dataset

from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++',  random_state = 0)

    kmeans.fit(features_data)

    wcss.append(kmeans.inertia_)

    

#Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia/WCSS') #within cluster sum of squares

plt.xticks(range(1,11))

plt.tight_layout()

plt.show()



#Using K=3 in the KMeans method to fit and predict the dataset



kmeans = KMeans(n_clusters = 3, init = 'k-means++',random_state = 0)

y_kmeans = kmeans.fit_predict(features_data)



y_kmeans
#Plotting the clusters



X_data = features_data.copy()

X_data["Class"] = y_kmeans



plt.scatter(X_data.SepalLengthCm[y_kmeans == 0], X_data.PetalLengthCm[y_kmeans == 0], s = 50, c = 'black', label = 'Iris-setosa')

plt.scatter(X_data.SepalLengthCm[y_kmeans == 1], X_data.PetalLengthCm[y_kmeans == 1], s = 50, c = 'red', label = 'Iris-versicolour')

plt.scatter(X_data.SepalLengthCm[y_kmeans == 2], X_data.PetalLengthCm[y_kmeans == 2], s = 50, c = 'purple', label = 'Iris-virginica')



#Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 50, c = 'blue', label = 'Centroids')
#Using Hierarchical clustering. 

#Data should be scaled first in hierarchical clustering

from sklearn.cluster import AgglomerativeClustering 

from sklearn.preprocessing import StandardScaler, normalize 

from sklearn.metrics import silhouette_score 

import scipy.cluster.hierarchy as shc 



scaler = StandardScaler() 

X_scaled = scaler.fit_transform(features_data) 



#Normalise the scaled data

X_normalized = normalize(X_scaled) 



# Converting the numpy array into a pandas DataFrame 

X_normalized_df = pd.DataFrame(X_normalized,columns = features_data.columns)



plt.figure(figsize =(8, 8)) 

plt.title('Visualising the data') 

Dendrogram = shc.dendrogram((shc.linkage(X_normalized, method ='ward'))) 

#Using Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering



hc_cluster = AgglomerativeClustering(n_clusters = 3) 

hc_predict = hc_cluster.fit_predict(X_normalized_df)

hc_predict

  

# Visualizing the clustering  

X_hier = X_normalized_df.copy()

X_hier["Class"] = hc_predict





plt.figure(figsize =(6, 6))

plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 0], X_hier.PetalLengthCm[X_hier.Class == 0], s = 100, c = 'red', label = 'Iris-versicolor')

plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 1], X_hier.PetalLengthCm[X_hier.Class == 1], s = 100, c = 'blue', label = 'Iris-setosa')

plt.scatter(X_hier.SepalLengthCm[X_hier.Class == 2], X_hier.PetalLengthCm[X_hier.Class == 2], s = 100, c = 'green', label = 'Iris-virginica')

plt.legend()

plt.title("Hierarchical") 

plt.xlabel("SepalLengthCm")

plt.ylabel("PetalLengthCm")

plt.show() 

k = [2,3,4,5]

silhouette_scores = [] 

def predictClusterNo(normal_df):

    for i in k:

        temp_cluster = AgglomerativeClustering(n_clusters = i)    

        silhouette_scores.append(silhouette_score(X_normalized_df,temp_cluster.fit_predict(X_normalized_df,cmap ='rainbow')))

    return 



predictClusterNo(X_normalized_df)

print(silhouette_scores)

plt.bar(k, silhouette_scores) 

plt.xlabel('Number of clusters', fontsize = 20) 

plt.ylabel('S(i)', fontsize = 20) 

plt.show() 

    

    