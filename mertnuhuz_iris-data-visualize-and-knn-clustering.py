import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from pandas.plotting import scatter_matrix

import mglearn
from sklearn.datasets import load_iris 

iris_dataset = load_iris()
print("Keys of iris set: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] +"\n...")
print('Target names: {}'.format(iris_dataset['target_names']))
print('Feature names: \n{}'.format(iris_dataset['feature_names']))
print('Shape of data: {}'.format(iris_dataset['data'].shape))
print('First five columns of data:\n{}'.format(iris_dataset['data'][:5]))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
iris_dataframe = pd.DataFrame(X_train, columns= iris_dataset.feature_names)



grr = scatter_matrix(iris_dataframe,

                        c=y_train,

                        figsize=(15,15),

                        marker="o",

                        hist_kwds={'bins':20},

                        s=60,

                        alpha=.8,

                        cmap=mglearn.cm3)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)

print(prediction, iris_dataset['target_names'][prediction])
y_pred = knn.predict(X_test)

print('test score: {:.2f}'.format(np.mean(y_pred == y_test)))
print('test score: {:.2f}'.format(knn.score(X_test, y_test)))
X_train, X_test, y_train, y_test = train_test_split(

        iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))