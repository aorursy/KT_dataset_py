import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



iris = pd.read_csv("../input/Iris.csv")
# Here is a peak of the data

iris.head()
X = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

y = iris["Species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
X_train.shape,X_test.shape
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=7)

model.fit(X_train, y_train) 
model.score(X_test, y_test)
neighbors = np.arange(1,9) # array([1, 2, 3, 4, 5, 6, 7, 8])

train_accuracy = np.empty(len(neighbors)) # array([......])

test_accuracy = np.empty(len(neighbors)) # array([......])
for i, n in enumerate(neighbors):

    model = KNeighborsClassifier(n_neighbors=n)

    model.fit(X_train, y_train) 

    train_accuracy[i] = model.score(X_train, y_train)

    test_accuracy[i] = model.score(X_test, y_test)
train_accuracy, test_accuracy
plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()