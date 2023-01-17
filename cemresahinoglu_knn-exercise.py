import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
data.info()
data.head()
data["class"] = [1 if i == "Abnormal" else 0 for i in data["class"]]
y = data["class"].values

x_ = data.drop(columns = ["class"])
x = ((x_ - np.min(x_)) / (np.max(x_) - np.min(x_)))
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)



knn.predict(x_test)
print(knn.score(x_test, y_test))
scores = []

for each in range(1, 15):

    knn2 = KNeighborsClassifier(n_neighbors= each)

    knn2.fit(x_train, y_train)

    scores.append(knn2.score(x_test, y_test))

    

plt.plot(range(1, 15), scores)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 13)

knn.fit(x_train, y_train)



knn.predict(x_test)
print(knn.score(x_test, y_test))