#Importing some necessary tools

import numpy as np

import pandas as pd

from numpy import genfromtxt
#Reading CSV files into panda

mall_data = pd.read_csv('../input/mall-customer/Mall_Customers.csv')
#Check for missing values (there are no)

mall_data.isnull().sum()
#Converting panda to Numpy NDArray

data = mall_data.to_numpy()
#K-means clustering

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn import metrics



#Using "elbow method" to find the best K

def E_M(data):

    #Excluding CustomerID from clustering 

    x = np.delete(data, [0], axis=1)

    

    #Replacing Male and Female with 1 and 0 respectively

    y = x[:,0]=='Male'

    y = y.astype(int)

    np.transpose(y)

    x[:,0] = np.copy(y)

    wcss = [] #Within Cluster Sum of Squares

    for i in range(1,4):

        kmeans = KMeans(n_clusters=i, random_state=0)

        kmeans.fit(x)

        wcss.append(kmeans.inertia_)

    plt.plot(range(1,4), wcss)

    plt.title('The Elbow Method')

    plt.xlabel('Number of Clusters')

    plt.ylabel('WCSS')

    plt.show()

    

    #It appears that the elbow is formed at 2

    K_M(x, 2)



def K_M(x, a):

    kmeans = KMeans(n_clusters=a, random_state=0)

    y_kmeans = kmeans.fit(x)

    

    #Assesing kmeans with silhouette metric

    sils = metrics.silhouette_score(x, y_kmeans.labels_, metric='euclidean')

    print(sils)

    

    #Scatter plot 1

    print("Annual Income vs Spending Score")

    plt.subplot(211)

    plt.xlabel('Annual Income')

    plt.ylabel('Spending Score')

    plt.scatter(x[:,2], x[:,3], c = y_kmeans.labels_, cmap='rainbow')



    #Scatter plot 2

    print("Annual Income vs Genre")

    plt.subplot(212)

    plt.xlabel('Annual Income')

    plt.ylabel('Genre')

    plt.scatter(x[:,2], x[:,0], c = y_kmeans.labels_, cmap='rainbow')

E_M(data)    

    