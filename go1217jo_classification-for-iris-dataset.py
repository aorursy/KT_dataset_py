from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset['DESCR'])
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib.colors import ListedColormap

X_train, X_test, y_train, y_test = train_test_split( 
    iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap= cm3)
import numpy as np
print(iris_dataset['feature_names'])
X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)
prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("            %s" % iris_dataset['target_names'][prediction])
y_pred = knn.predict(X_test)
accuracy = np.mean(y_test == y_pred)
print("accuracy : %.2f" % accuracy)