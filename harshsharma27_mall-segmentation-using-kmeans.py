import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/mall-customers/Mall_Customer_Dataset.csv')
df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()
df.corr()['Spending Score (1-100)'].sort_values(ascending=False)
sns.pairplot(df)
sns.set_style('darkgrid')

plt.figure(figsize=(12,6))

plt.scatter(x='Age',y='Annual Income (k$)',data=df)

plt.xlabel('Age')

plt.ylabel('Annual Income in k$')
X=df.iloc[:,[3,4]].values
X.shape
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3,init='k-means++',random_state=0)

y_predicted=km.fit_predict(X)
y_predicted
df['Cluster']=y_predicted
df.head()
km.cluster_centers_
plt.scatter(X[y_predicted == 0, 0], X[y_predicted == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_predicted == 1, 0], X[y_predicted == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_predicted == 2, 0], X[y_predicted == 2, 1], s = 100, c = 'green', label = 'Cluster 3')



#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids',marker='*')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
error=[]

for i in range(1,20):

    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)

    kmeans.fit(X)

    error.append(kmeans.inertia_)
plt.plot(range(1,20),error)

plt.xlabel('K Values')

plt.ylabel('Error')
kmeans_new=KMeans(n_clusters=5,init='k-means++', random_state=0)

y_kmeans=kmeans_new.fit_predict(X)
y_kmeans
df['Cluster']=y_kmeans
df.head()
plt.figure(figsize=(12,6))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans_new.cluster_centers_[:, 0], kmeans_new.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids',marker='*')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
#Cluster 1 (Red Color) -> Average Income , average spending

#cluster 2 (Blue Color) -> Less income,high spending 

#cluster 3 (Green Color) -> earning high and also spending high [Potential Target Customers]

#cluster 4 (cyan Color) -> earning less but spending less

#Cluster 5 (magenta Color) -> Earning High , spending less
"""

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')



here plt.scatter(x,y)=x=y_kmeans==0,0 ->first 0 indicates= cluster label 0 &second 0 indicates feature number=Annual Income

                    y=y_kmeans=0,1-> first 0 indicates cluster label 0 & second 1 indicates feature number 1=Spending score

"""