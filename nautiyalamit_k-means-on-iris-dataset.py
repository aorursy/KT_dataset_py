#iris_dataset = pd.read_csv('./Iris.csv')

from sklearn.datasets import load_iris

import pandas as pd

import numpy as np 

iris_dataset= load_iris()
##loaded the iris dataset 

##now lets print the iris dataset 

iris_dataset
#The value of the key target_names is an array of strings, containing the species of

#flower that we want to predict:

print(iris_dataset['feature_names'])
from pandas.plotting import scatter_matrix



# create dataframe from data in X_train

# label the columns using the strings in iris_dataset.feature_names

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)



iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

print(iris_dataframe)

# create a scatter matrix from the dataframe, color by y_train

grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)

print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])

print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))

print("Predicted target name: {}".format(

iris_dataset['target_names'][prediction]))
y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))