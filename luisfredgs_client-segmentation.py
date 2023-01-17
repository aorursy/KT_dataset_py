import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
df = pd.read_csv("../input/dataset.csv")

df.columns = ["CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"]

df.head(5)
df.shape
df.isnull().sum()
X = df.iloc[:, [3,4]].values

X
scaler = StandardScaler()

X = scaler.fit_transform(X)

X
df.iloc[:, [3,4]] = X
plt.scatter(X[:, 0], X[:, 1])

plt.show()
sse = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i, init="k-means++")

    kmeans.fit(X)

    sse.append(kmeans.inertia_)

    

plt.plot(range(1,11), sse)

plt.title("Elbow")

plt.xlabel("number of clusters")

plt.ylabel("sse")

plt.show()
n_clusters = 5

model = KMeans(n_clusters=n_clusters, init="k-means++")

pred = model.fit_predict(X)
plt.figure(figsize=(20,10))

for i in range(0, n_clusters):

    plt.scatter(X[pred == i, 0], X[pred == i, 1], s=50, label="Cluster %d" % i)

    

plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s = 100, c = 'black', label='centroids')

plt.title("Clusters")

plt.xlabel("Annual Income")

plt.ylabel("Spending_Score")

plt.legend()

plt.show()
for row in X[pred == 0]:

    print(df[(df.iloc[:, 3] == row[0]) & (df.iloc[:,4] == row[1])]['CustomerID'].values)