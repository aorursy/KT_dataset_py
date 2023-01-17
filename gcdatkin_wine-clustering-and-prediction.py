import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



from sklearn.neural_network import MLPClassifier
data = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
data
data.isnull().sum()
data.dtypes
data['quality'].unique()
y = data['quality']

X = data.drop('quality', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
kmeans = KMeans(n_clusters=6)

kmeans.fit(X)
clusters = kmeans.predict(X)

clusters
pca = PCA(n_components=2)



reduced_X = pd.DataFrame(pca.fit_transform(X), columns=["PC1", "PC2"])

reduced_X
reduced_X['cluster'] = clusters

reduced_X
plt.figure(figsize=(14, 10))



plt.scatter(reduced_X[reduced_X['cluster'] == 0].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 0].loc[:, 'PC2'], color='slateblue')

plt.scatter(reduced_X[reduced_X['cluster'] == 1].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 1].loc[:, 'PC2'], color='springgreen')

plt.scatter(reduced_X[reduced_X['cluster'] == 2].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 2].loc[:, 'PC2'], color='indigo')

plt.scatter(reduced_X[reduced_X['cluster'] == 3].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 3].loc[:, 'PC2'], color='teal')

plt.scatter(reduced_X[reduced_X['cluster'] == 4].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 4].loc[:, 'PC2'], color='lightcoral')

plt.scatter(reduced_X[reduced_X['cluster'] == 5].loc[:, 'PC1'], reduced_X[reduced_X['cluster'] == 5].loc[:, 'PC2'], color='gold')



plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], color='black', marker='x', s=300)



plt.show()
reduced_centers = pca.transform(kmeans.cluster_centers_)

reduced_centers
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = MLPClassifier(hidden_layer_sizes=(256, 256), max_iter=500)



model.fit(X_train, y_train)
print(f"Model Accuracy: {model.score(X_test, y_test)}")