# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the dataset

import pandas as pd

dataset = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
# Displaying the head of out dataset

dataset.head()
# Displaying the datatype and shape of the dataset



print(dataset.shape)

dataset.dtypes
# Displaying the describe statistical info of each attribute

dataset.describe()
# Histogram visualisation for age column and know what kind of distribution  it is ?



import seaborn as sb

sb.distplot(dataset['Age'])
# Histogram visualisation for Annual income column and know what kind of distribution  it is ?



sb.distplot(dataset['Annual Income (k$)'])
# Histogram visualiation for Spending Score column and to know what kind of distribution it is?



sb.distplot(dataset['Spending Score (1-100)'])
# Visualisation correlation coefficient of each attribute.



corr_value=dataset.corr()

sb.heatmap(corr_value,square=True)
# Displying any empty or null values in our dataset

dataset.info()
# Displying the empty or null value in our dataset to understand better how many missing cells there in each attribute



dataset.isna().sum()
# Dropping CustomerID column 



dataset=dataset.drop(['CustomerID'],axis=1)

dataset.head()
# Encoding the Gender column from categorical value into numerical value



dataset['Gender'].unique()
dataset['Gender']=dataset['Gender'].map({'Male':0,'Female':1})

dataset['Gender'].unique()
dataset.head()
# Feature Split

x=dataset.values



print(x[:5,:])
# Feature Scale



from sklearn.preprocessing import MinMaxScaler

minmaxscaler=MinMaxScaler()

x=minmaxscaler.fit_transform(x)

print(x[:5,:])
# Elbow Method



seed=5



from sklearn.cluster import KMeans

import matplotlib.pyplot as plt



wcss=[]

# n_init ----- Number of kmeans will run with different init centroids

# max_iter------ Max Number of iterations to define that the final clusters

# init='k-means++' ---- random initlization to handle random intialization trap

for i in range(1,11):

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=500,n_init=20,random_state=seed)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,11),wcss)

plt.title("Elbow method")

plt.xlabel("No.of Clusters")

plt.ylabel('WCSS')

plt.show()
# K-Means Cluster Algorithm

kmeans=KMeans(n_clusters=4,init='k-means++',random_state=seed,max_iter=500,n_init=20)

y_kmeans=kmeans.fit_predict(x)



# Predicting the Customers with different segments

print(y_kmeans)

plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster 1')

plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='Cluster 2')

plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='Cluster 3')

plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='cyan',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,color='yellow',label='Centroid')

plt.title("Cluster Clients")

plt.xlabel('Annual income')

plt.ylabel('spending score')

plt.legend()

plt.show()
# Dendo Gram plot is used to find optimal number of cluster..



import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(x,method='ward'))

plt.title('Dendogram')

plt.xlabel('customers')

plt.ylabel('Eulidean distance')

plt.show()
# Hierarchical Clustering Algorithm to the mall dataset

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=4)

hc.fit(x)


# Predict the cluster categories based on mall dataset

y_hc=hc.fit_predict(x)

print(y_hc)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster 1')

plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='Cluster 2')

plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='Cluster 3')

plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='cyan',label='cluster 4')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,color='yellow',label='Centroid')

plt.title("Cluster Clients")

plt.xlabel('Annual income')

plt.ylabel('spending score')

plt.legend()

plt.show()