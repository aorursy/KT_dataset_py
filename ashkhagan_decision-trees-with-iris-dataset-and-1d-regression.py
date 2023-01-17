from sklearn.datasets import load_iris #import dataset

iris = load_iris()                     #loading the data

iris                                   #print the data
!pip install pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pydotplus

from IPython.display import Image



clf = DecisionTreeClassifier().fit(iris.data, iris.target)

dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True,

                                feature_names=iris.feature_names,  

                                class_names=['Versicolor','Setosa','Virginica'])

graph = pydotplus.graph_from_dot_data(dot_data)  

image=graph.create_png()

graph.write_png("kmc_dt.png")

Image(filename="kmc_dt.png")
print(__doc__)



# Import the necessary modules and libraries

import numpy as np

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt



# Create a random dataset

rng = np.random.RandomState(1)

X = np.sort(5 * rng.rand(80, 1), axis=0)

y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - rng.rand(16))



# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=2)

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X, y)

regr_2.fit(X, y)



# Predict

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]

y_1 = regr_1.predict(X_test)

y_2 = regr_2.predict(X_test)



# Plot the results

plt.figure()

plt.scatter(X, y, s=20, edgecolor="black",

            c="darkorange", label="data")

plt.plot(X_test, y_1, color="cornflowerblue",

         label="max_depth=2", linewidth=1)

plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)

plt.xlabel("data")

plt.ylabel("target")

plt.title("Decision Tree Regression")

plt.legend()

plt.show()