# Walking through the Iris Classification (k-nearest neighbors) example from Introduction to Machine Learning with Python by Andreas Muller & Sarah Guido 

from sklearn.datasets import load_iris
iris_dataset = load_iris()

import numpy as np # linear algebrad
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# print out the description of iris dataset
print(iris_dataset['DESCR'][:193] + "\n...")
# print out the target names of the species to predict
print("Target names: {}".format(iris_dataset['target_names']))
# print out the feature names
print("Feature names: {}".format(iris_dataset['feature_names']))
# print out the data type of the iris_dataset.data
print("Type of data: {}".format(type(iris_dataset['data'])))
# print the shape of the dataset (150,4) - "The shape of the data array is the number of samples multiplied by the number of features."
print("Shape of data: {}".format(iris_dataset['data'].shape))
# print the first five rows of the dataset
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))
# print the datatype of the target dataset (iris_dataset.target) - the target represents the actual pre-defined species of the flower, the target varaible
print("Type of target: {}".format(type(iris_dataset['target'])))
# print the shape of the target dataset
print("Shape of target: {}".format(iris_dataset['target'].shape))
# print the target data
print("Target: \n{}".format(iris_dataset['target']))
# import train_test_split to segment data into training/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8)
# import the k-nearest neighbors classifier from sci-kit learn
from sklearn.neighbors import KNeighborsClassifier
# Instantiate the KNeighborsClassifier with a n_neighbors value of 1
knn = KNeighborsClassifier(n_neighbors=1)
# build the model from the training set
knn.fit(X_train, y_train)
# Create a new sample and use the model built above to predict the species
x_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(x_new.shape))
# make a predition based on the above sample
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
# run the test data set through the model to determine predictions =
y_pred = knn.predict(X_test)
print("Test set predictions: \n {}".format(y_pred))
# test the accuracy of the model by using np.mean function
print("Test set score: {:.2f}".format(np.mean(y_pred==y_test)))
# test the accuract of the model using the score function
print("Test set score: {:2f}".format(knn.score(X_test, y_test)))