# for basic mathematics operation 

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



# for path

import os

print(os.listdir('../input/'))
# importing the dataset

%time data = pd.read_csv('../input/Mall_Customers.csv')



print(data.shape)
# getting the head of the data



data.head()
# describing the data



data.describe()
# checking if there is any NULL data



data.isnull().any().any()
plt.rcParams['figure.figsize'] = (18, 8)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(data['Annual Income (k$)'])

plt.title('Distribution of Annual Income', fontsize = 20)

plt.xlabel('Range of Annual Income')

plt.ylabel('Count')





plt.subplot(1, 2, 2)

sns.set(style = 'whitegrid')

sns.distplot(data['Age'], color = 'red')

plt.title('Distribution of Age', fontsize = 20)

plt.xlabel('Range of Age')

plt.ylabel('Count')

plt.show()
labels = ['Female', 'Male']

size = data['Gender'].value_counts()

colors = ['lightgreen', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (7, 7)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('A pie chart Representing the Gender')

plt.axis('off')

plt.legend()

plt.show()
plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['Age'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Distribution of Annual Income', fontsize = 20)
plt.rcParams['figure.figsize'] = (20, 8)

sns.countplot(data['Spending Score (1-100)'], palette = 'copper')

plt.title('Distribution of Spending Score', fontsize = 20)
sns.pairplot(data)
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data', fontsize = 20)
#  Gender vs Spendscore



plt.rcParams['figure.figsize'] = (18, 7)

sns.boxenplot(data['Gender'], data['Spending Score (1-100)'], palette = 'Blues')

plt.title('Gender vs Spending Score', fontsize = 20)
plt.rcParams['figure.figsize'] = (18, 7)

sns.violinplot(data['Gender'], data['Annual Income (k$)'], palette = 'rainbow')

plt.title('Gender vs Spending Score', fontsize = 20)
plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['Gender'], data['Age'], palette = 'Purples', size = 10)

plt.title('Gender vs Spending Score', fontsize = 20)
data.columns


x = data['Annual Income (k$)']

y = data['Age']

z = data['Spending Score (1-100)']



sns.lineplot(x, y, color = 'blue')

sns.lineplot(x, z, color = 'pink')

plt.title('Annual Income vs Age and Spending Score', fontsize = 20)

plt.legend()

plt.show()
x = data.iloc[:, [3, 4]].values



print(x.shape)
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

  km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

  km.fit(x)

  wcss.append(km.inertia_)

  

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)



plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()



import scipy.cluster.hierarchy as sch



dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('Customers')

plt.ylabel('Ecuclidean Distance')

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('Annual Income')

plt.ylabel('Spending Score')

plt.legend()

plt.show()
data.columns
x = data.iloc[:, [2, 4]].values

x.shape
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

  kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

  kmeans.fit(x)

  wcss.append(kmeans.inertia_)



plt.rcParams['figure.figsize'] = (15, 5)

plt.plot(range(1, 11), wcss)

plt.title('K-Means Clustering(The Elbow Method)', fontsize = 20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.show()
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')



plt.xlabel('Age')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()