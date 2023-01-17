%matplotlib inline

import numpy as np

import pandas as pd



from sklearn import preprocessing

from sklearn.model_selection import train_test_split
iris = pd.read_csv("../input/Iris.csv")

iris.head()
iris.columns
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
le = preprocessing.LabelEncoder()

y = le.fit_transform(iris[['Species']]); y
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))

print("y_test shape: {}".format(y_test.shape))
# create dataframe from data in X_train

# label the columns using the strings in iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

# create a scatter matrix from the dataframe, color by y_train

pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',

                           hist_kwds={'bins': 20}, s=60, alpha=.8)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1, 0.2]])

print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)

print("Prediction: {}".format(prediction))

print("Predicted target name: {}".format(le.inverse_transform(prediction)))
y_pred = knn.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))