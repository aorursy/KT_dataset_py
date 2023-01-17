import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/store-retail-analysis/Store_Customers.csv')

df.columns=['CustomerID', 'Gender', 'Age', 'Annual Income (k$)',

       'Spending Score (1-100)']

df
df.shape
df.describe()
df.dtypes
df.isnull().sum()
df.columns
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)

    sns.distplot(df[x] , bins = 20)

    plt.title('Distplot of {}'.format(x))

plt.show()
plt.figure(1 , figsize = (15 , 5))

sns.countplot(y = 'Gender' , data = df)

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

        n += 1

        plt.subplot(3 , 3 , n)

        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

        sns.regplot(x = x , y = y , data = df)

        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )

plt.show()
plt.figure(1 , figsize = (15 , 6))

for gender in ['Male' , 'Female']:

    sns.regplot(x = 'Age' , y = 'Annual Income (k$)' , data = df[df['Gender'] == gender]  , label = gender,fit_reg=False)

    

plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 

plt.title('Age vs Annual Income w.r.t Gender')

plt.legend()

plt.show()
plt.figure(1 , figsize = (15 , 7))

n = 0 

for cols in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:

    n += 1 

    plt.subplot(1 , 3 , n)

    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)

    sns.violinplot(x = cols , y = 'Gender' , data = df , palette = 'vlag')

    sns.swarmplot(x = cols , y = 'Gender' , data = df)

    plt.ylabel('Gender' if n == 1 else '')

    plt.title('Boxplots & Swarmplots' if n == 2 else '')

plt.show()
X = df.iloc[:, [3, 4]].values
# Using the elbow method to find the optimal number of clusters

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , wcss , 'o')

plt.plot(np.arange(1 , 11) , wcss , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')

plt.show()
# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)
# Visualizing the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = '#e01470', label = 'Standard Customers')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'darkblue', label = 'Careful Customers')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = '#eb4034', label = 'Sensible Customers')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = '#4cadad', label = 'Careless Customers')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = '#54eb09', label = 'Target Customers')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = '#232414', alpha=0.6,label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
#Using Dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method='ward'))

plt.title('Dendogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean Distances')

plt.show()
#Fitting heirarchical clustering to the store dataset

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)
#Visualizing the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = '#e01470', label = 'Standard Customers')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'darkblue', label = 'Careful Customers')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = '#eb4034', label = 'Sensible Customers')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = '#4cadad', label = 'Careless Customers')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = '#54eb09', label = 'Target Customers')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
!pip install apyori
# Data Preprocessing

dataset = pd.read_csv('../input/market-basket-analysis-dataset/Market_Basket_Optimisation.csv', header = None)

transactions = []

for i in range(0, 7501):

    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
# Training Apriori on the dataset

from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#Top 5 grouped products

results = list(rules)

for i in range(0,5):

    print(results[i])

    print('**********')
# Visualizing all the results

for i in results:

    print(i)

    print('**********')