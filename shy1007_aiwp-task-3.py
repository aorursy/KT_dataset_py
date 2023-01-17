import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth

K = 15
col_names = ["customer_id","gender","age","annual_income","spending_score"]
data = pd.read_csv("../input/mall-customers/Mall_Customers.csv",names=col_names,header=0)
data = data.sample(frac=1)
data.head()
data.shape
gender_data = data.iloc[:,[1,4]]
gender_data = gender_data.replace(to_replace="Female",value=1)
gender_data = gender_data.replace(to_replace="Male",value=0)
gender_data.head()



total_females = sum([1 for i in gender_data["gender"] if i == 1 ])
total_males = sum([1 for i in gender_data["gender"] if i == 0 ])
gender_fig = plt.figure()
ax = gender_fig.add_axes([0,0,1,1])
genders = ["Female","Male"]
frequency = [total_females,total_males]
ax.bar(genders,frequency)
ax.set_xlabel('Gender')
ax.set_ylabel('Frequency')
plt.show()

gender_inertia = []

for k in range(1,K):
  gender_kmean = KMeans(n_clusters=k,init = 'k-means++')
  gender_kmean.fit(gender_data.values)
  gender_inertia.append(gender_kmean.inertia_)
plt.plot(range(1,K), gender_inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distance of points with centroid')
plt.show()
X = gender_data.values
gender_k = 3 #from graph above
gender_kmean = KMeans(n_clusters = gender_k, init = 'k-means++')
gender_y = gender_kmean.fit(gender_data.values)
X = gender_data.values
plt.scatter(X[:, 0], X[:, 1], s = 100, c = gender_y.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(gender_kmean.cluster_centers_[:, 0], gender_kmean.cluster_centers_[:, 1], s = 100, c = 'black')
plt.title('Clusters of customers')
plt.xlabel('Gender')
plt.ylabel('Spending Score (1-100)')
plt.show()
gender_afprop = AffinityPropagation()
gender_afprop.fit(X)
cluster_centers = gender_afprop.cluster_centers_indices_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = gender_afprop.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(gender_afprop.cluster_centers_[:, 0], gender_afprop.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Gender")
plt.ylabel('Spending Score (1-100)')
plt.show()
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10)
gender_ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
cluster_centers = gender_ms.cluster_centers_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = gender_ms.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(gender_ms.cluster_centers_[:, 0], gender_ms.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Gender")
plt.ylabel('Spending Score (1-100)')
plt.show()
age_data = data.iloc[:,[2,4]]
age_X = age_data.values
age_data.head()

plt.scatter(age_X[:,0],age_X[:,1],s=100)
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.title("spending_score-age scatter plot")
plt.show()

age_inertia = []
for k in range(1,K):
  age_kmean = KMeans(n_clusters=k,init = 'k-means++')
  age_kmean.fit(age_X)
  age_inertia.append(age_kmean.inertia_)
plt.plot(range(1,K), age_inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distance of points with centroid')
plt.show()
X = age_data.values
age_k = 4 #from graph above
age_kmean = KMeans(n_clusters = age_k, init = 'k-means++')
age_y = age_kmean.fit(age_data.values)
X = age_data.values
plt.scatter(X[:, 0], X[:, 1], s = 100, c = age_y.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(age_kmean.cluster_centers_[:, 0], age_kmean.cluster_centers_[:, 1], s = 100, c = 'black')
plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()
age_afprop = AffinityPropagation()
age_afprop.fit(X)
cluster_centers = age_afprop.cluster_centers_indices_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = age_afprop.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(age_afprop.cluster_centers_[:, 0], age_afprop.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Age")
plt.ylabel('Spending Score (1-100)')
plt.show()
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=10)
age_ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
cluster_centers = age_ms.cluster_centers_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = age_ms.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(age_ms.cluster_centers_[:, 0], age_ms.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Age")
plt.ylabel('Spending Score (1-100)')
plt.show()
annual_income_data = data.iloc[:,[3,4]]
annual_income_X = annual_income_data.values
annual_income_data.head()
plt.scatter(annual_income_X[:,0],annual_income_X[:,1],s=100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("spending_score-annual_income scatter plot")
plt.show()
annual_income_inertia = []
for k in range(1,K):
  annual_income_kmean = KMeans(n_clusters=k,init = 'k-means++')
  annual_income_kmean.fit(annual_income_X)
  annual_income_inertia.append(annual_income_kmean.inertia_)
plt.plot(range(1,K), annual_income_inertia)
plt.title('Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Distance of points with centroid')
plt.show()
X = annual_income_data.values

annual_income_k = 5 #from graph above
annual_income_kmean = KMeans(n_clusters = annual_income_k, init = 'k-means++')
annual_income_y = annual_income_kmean.fit(X)
plt.scatter(X[:, 0], X[:, 1], s = 100, c = annual_income_y.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(annual_income_kmean.cluster_centers_[:, 0], annual_income_kmean.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.show()
annual_income_afprop = AffinityPropagation()
annual_income_afprop.fit(X)
cluster_centers = annual_income_afprop.cluster_centers_indices_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = annual_income_afprop.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(annual_income_afprop.cluster_centers_[:, 0], annual_income_afprop.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.show()
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=30)
annual_income_ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
cluster_centers = annual_income_ms.cluster_centers_
n_clusters = len(cluster_centers)

print(f"total number of cluster = {n_clusters}")
plt.scatter(X[:, 0], X[:, 1], s = 100, c = annual_income_ms.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.scatter(annual_income_ms.cluster_centers_[:, 0], annual_income_ms.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel("Annual Income (k$)")
plt.ylabel('Spending Score (1-100)')
plt.show()