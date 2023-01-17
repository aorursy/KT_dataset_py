#import Libraries

import pandas as pd

import matplotlib.pyplot as plt
#import th data set

df = pd.read_csv(r'../input/Mall_Customers.csv')

df.head()

#select the row having annual income and Spending Score

Y= df.iloc[:, [3, 4]]

Y.head()
#Use elbow method to find the optimum number of Cluster for given data set

from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(Y)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()



#As you can see below once number of cluster reach to 5 then value of WCSS decreases at very slow rate so optimum

# number of cluster is Five

# to know more about Elbow Method and how to calculate WCSS send me an email at anoopchaudhary@gmail.com
# Fitting K-Means to the dataset

X = df.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)
# Visualising the clusters

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1],s=100,c='red', label='average')

plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1],s=100,c='Blue', label='Carefull')

plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1],s=100,c='gray', label='poor')

plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1],s=100,c='green', label='careless')

plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1],s=100,c='cyan', label='target')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroid')

plt.title('Cluster representation of data')

plt.legend()

plt.xlabel('Anuall income')

plt.ylabel('shoping score')

plt.show()