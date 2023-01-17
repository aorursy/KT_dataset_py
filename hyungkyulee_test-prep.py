from sklearn.datasets import load_iris

import numpy as np
iris = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state = 0)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1)
print (knn)
knn.fit(X_train, y_train)
knn.predict([[10, 3, 5, 1]])