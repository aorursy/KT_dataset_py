#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the Iris dataset with pandas
dataset = pd.read_csv('../input/Iris.csv')
x = dataset.iloc[:, [1, 2, 3, 4]].values
label = dataset.iloc[:,0].values
print("Input Data and Shape")
print(x.shape)
#y  = x.head(50) #khai bao them bien de thay doi du lieu
print(type(x))
#Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = [] # khai bao list danh sach array

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 50, n_init = 10, random_state = 10)
    kmeans.fit(x) # ap model len du lieu
    wcss.append(kmeans.inertia_) # khai bao trong so loi vaf luu loi tai moi lan chay tai wcss = 
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('tile bieu do')
plt.xlabel('cum')
plt.ylabel('SSE') #within cluster sum of squares
plt.show()
#Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 20, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
# output = y_kmeans.predict(x)
# print(output)
# y_kmeans.head()
dataframe = pd.DataFrame({"label":label,  "cluster":y_kmeans})
dataframe.head(150)
# print(y_kmeans)

#Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'versicolor')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'virginica')
#Plotting the centroids of the clusters 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroide')
plt.legend()
plt.show()