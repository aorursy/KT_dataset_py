import numpy as np

import pandas as pd

from sklearn import datasets, neighbors

from sklearn.neighbors import KNeighborsClassifier



iris = datasets.load_iris()

iris_data = iris.data

iris_labels = iris.target



iris_df = pd.DataFrame.from_records(iris.data)

iris_df['target'] = iris.target

iris_df = iris_df.replace({'target': 0}, 'Iris-Setosa')

iris_df = iris_df.replace({'target': 1}, 'Iris-Versicolor')

iris_df = iris_df.replace({'target': 2}, 'Iris-Virginica')



iris_df
x=iris_df.iloc[1:,:3] #features

y=iris_df.iloc[1:,4:] #target labels (Iris-Setosa, Iris-Versicolor, Iris-Virginica)



neigh=KNeighborsClassifier(n_neighbors=4)

neigh.fit(iris_df.iloc[:,:4],iris_df["target"])



testSet = [[4.8, 2.5, 5.3, 2.4]]

test = pd.DataFrame(testSet)



print("predicted:",neigh.predict(test))

print("neighbors",neigh.kneighbors(test))