import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data
data.drop('CustomerID', axis=1, inplace=True)
encoder = LabelEncoder()

data['Gender'] = encoder.fit_transform(data['Gender'])



gender_mappings = {index: label for index, label in enumerate(encoder.classes_)}

gender_mappings
scaler = StandardScaler()

scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
scaled_data
max_clusters = 50
kmeans_tests = [KMeans(n_clusters=i, n_init=10) for i in range(1, max_clusters)]

inertias = [kmeans_tests[i].fit(scaled_data).inertia_ for i in range(len(kmeans_tests))]
plt.figure(figsize=(7, 5))

plt.plot(range(1, max_clusters), inertias)

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.title("Choosing the Number of Clusters")

plt.show()
kmeans = KMeans(n_clusters=10, n_init=10)

kmeans.fit(scaled_data)
clusters = kmeans.predict(scaled_data)

clusters
pca = PCA(n_components=2)



reduced_data = pd.DataFrame(pca.fit_transform(scaled_data), columns=['PC1', 'PC2'])
reduced_data
kmeans.cluster_centers_
reduced_centers = pca.transform(kmeans.cluster_centers_)

reduced_centers
reduced_data['cluster'] = clusters
reduced_data
plt.figure(figsize=(14, 10))



plt.scatter(reduced_data[reduced_data['cluster'] == 0].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 0].loc[:, 'PC2'], color='red')

plt.scatter(reduced_data[reduced_data['cluster'] == 1].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 1].loc[:, 'PC2'], color='blue')

plt.scatter(reduced_data[reduced_data['cluster'] == 2].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 2].loc[:, 'PC2'], color='yellow')

plt.scatter(reduced_data[reduced_data['cluster'] == 3].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 3].loc[:, 'PC2'], color='orange')

plt.scatter(reduced_data[reduced_data['cluster'] == 4].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 4].loc[:, 'PC2'], color='cyan')

plt.scatter(reduced_data[reduced_data['cluster'] == 5].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 5].loc[:, 'PC2'], color='magenta')

plt.scatter(reduced_data[reduced_data['cluster'] == 6].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 6].loc[:, 'PC2'], color='brown')

plt.scatter(reduced_data[reduced_data['cluster'] == 7].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 7].loc[:, 'PC2'], color='pink')

plt.scatter(reduced_data[reduced_data['cluster'] == 8].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 8].loc[:, 'PC2'], color='green')

plt.scatter(reduced_data[reduced_data['cluster'] == 9].loc[:, 'PC1'], reduced_data[reduced_data['cluster'] == 9].loc[:, 'PC2'], color='purple')



plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='x', s=300)



plt.xlabel("PC1")

plt.ylabel("PC2")



plt.show()