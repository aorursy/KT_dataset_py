import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
dataset = pd.read_csv("../input/mall-customer-data/Mall_Customers.csv")

dataset.head(20)
dataset.rename(columns={"Genre":"Gender"}, inplace = True)

dataset
# info



dataset.info()
dataset.columns
dataset.describe()
df_gen  = dataset['Gender']

df_gen.value_counts().plot(kind = 'bar', rot = 0)
plt.figure(figsize = (10,6))

df_gen.value_counts().plot(kind = 'pie', legend = 1,autopct = '%0.2f%%', explode = [0,0.1], shadow = True, colors = ['lightblue', 'orange'])

plt.title('Gender', fontsize = 20)



plt.show()
plt.figure(figsize = (18,8))

sns.countplot(dataset['Age'], palette = 'hsv')
plt.figure(figsize  = (24,8))

sns.countplot(dataset['Annual Income (k$)'], palette = 'rainbow')

plt.figure(figsize = (18,8))

sns.countplot(dataset['Spending Score (1-100)'], palette = 'rainbow')
sns.pairplot(dataset)
dataset.corr()
sns.heatmap(dataset.corr(), annot  = True, cmap = 'RdYlGn')
X = dataset.iloc[:,[3,4]].values

print(X)
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)



plt.figure(figsize = (12,8))

plt.plot(range(1,11), wcss)

plt.title("Elbow method")

plt.xlabel("No. of Clusters")

plt.ylabel("wcss")

plt.show()





kmeans = KMeans(n_clusters = 5, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
plt.figure(figsize = (10,8))

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100 , c = "red", label = "Standard")

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100 , c = "blue", label = "Careless")

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100 , c = "green", label = "Target")

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100 , c = "cyan", label = "Sensible")

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100 , c = "magenta", label = "Careful")



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Centroids" )



plt.title("Clusters of clients")

plt.xlabel("Annual Income (K$)")

plt.ylabel("Spending-Score(1-100)")

plt.legend()

plt.show()
