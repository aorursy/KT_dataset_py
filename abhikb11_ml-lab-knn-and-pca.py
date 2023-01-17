import numpy as np
from sklearn import datasets, neighbors, linear_model, tree
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_olivetti_faces
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from time import time

faces_data = fetch_olivetti_faces()

n_samples, height, width = faces_data.images.shape
X = faces_data.data
n_features = X.shape[1]
Y = faces_data.target

number_of_samples = len(Y)

random_indices = np.random.permutation(number_of_samples)
#Training set
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)



#KNN
model = neighbors.KNeighborsClassifier(n_neighbors = 5) # K = 5
model.fit(x_train, y_train)
print(model.score(x_test,y_test))


#PCA
faces_data = fetch_olivetti_faces()

n_samples, height, width = faces_data.images.shape
X = faces_data.data
n_features = X.shape[1]
Y = faces_data.target
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42)

#Reduce the dimensionality of the feature space
n_components = 150

#Finding the top n_components principal components in the data
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

#Find the eigen-vectors of the feature space
eigenfaces = pca.components_.reshape((n_components, height, width))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


knn_classifier = KNeighborsClassifier(n_neighbors = 5)
knn_classifier.fit(X_train_pca, Y_train)
print(knn_classifier.score(X_test_pca,Y_test))