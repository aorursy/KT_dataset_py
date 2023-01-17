import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
%matplotlib inline
iris_data = load_iris()
print("Look inside dataset: {}".format(iris_data.keys()))
print("Description of data set: {}".format(iris_data['DESCR']))
print("Feature names: {}".format(iris_data['feature_names']))
print("Target array: {}".format(iris_data['target']))
X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)
print("X_train: {}".format(X_train.shape))
print("y_train: {}".format(y_train.shape))
print("X_test: {}".format(X_test.shape))
print("y_test: {}".format(y_test.shape))
pd_iris = pd.DataFrame(X_train, columns=iris_data['feature_names'])
pd.plotting.scatter_matrix(pd_iris, c=y_train, figsize=(15, 15), 
                           marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Predicted: {}".format(prediction))
print("Predicted name: {}".format(iris_data['target_names'][prediction]))
# Calculate accuracy of the model by comparing predictions from 
# test data set to test target values
y_pred = knn.predict(X_test)
print("Model accuracy: {:.2f}".format(np.mean(y_pred == y_test)))