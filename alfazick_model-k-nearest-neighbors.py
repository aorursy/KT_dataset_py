# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:200] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

iris_dataset['data'],iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
# create dataframe from data in X_train

# label the columns using the strings in iris_dataset.features_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train

grr = pd.scatter_matrix(iris_dataframe, c = y_train, marker='o',figsize=(15,15),

                       hist_kwds={'bins':20}, s = 60, alpha= .8)
# Building First Model: k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5,2.9,1,0.2]])

print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))

print("Predicted target name: {}".format(

iris_dataset['target_names'][prediction]))
# Evaluating the Model
y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# Summary code for all model
X_train, X_test, y_train, y_test = train_test_split(

iris_dataset['data'], iris_dataset['target'], random_state=0)



knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)



print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))