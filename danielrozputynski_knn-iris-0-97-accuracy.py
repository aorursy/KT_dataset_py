import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

iris = pd.read_csv('../input/Iris.csv')
iris.head()
iris.describe()
iris.info()
iris_matrix = iris.drop('Id', axis=1)

sns.pairplot(iris_matrix, hue='Species', markers='p')

plt.show()
plt.figure(figsize=(20, 15))



plt.subplot(2,2,1)



sns.boxplot(x="Species", y="SepalLengthCm", data=iris)

sns.swarmplot(x="Species", y="SepalLengthCm", data=iris, color=".15")

plt.title('SepalLengthCm', fontsize = 20)



plt.subplot(2,2,2)

sns.boxplot(x="Species", y="SepalWidthCm", data=iris)

sns.swarmplot(x="Species", y="SepalWidthCm", data=iris, color=".15")

plt.title('SepalWidthCm', fontsize = 20)



plt.subplot(2,2,3)



sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

sns.swarmplot(x="Species", y="PetalLengthCm", data=iris, color=".15")

plt.title('PetalLengthCm', fontsize = 20)



plt.subplot(2,2,4)

sns.boxplot(x="Species", y="PetalWidthCm", data=iris)

sns.swarmplot(x="Species", y="PetalWidthCm", data=iris, color=".15")

plt.title('PetalWidthCm', fontsize = 20)



plt.tight_layout()

plt.show()
irisArray = iris.drop(['Id', 'Species'], axis=1).values

irisArray.shape
IrisN = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,

        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

IrisN.shape
X_train, X_test, y_train, y_test = train_test_split(irisArray, IrisN, random_state=0)
neighbors = np.arange(1, 30)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    train_accuracy[i] = knn.score(X_train, y_train)

    test_accuracy[i] = knn.score(X_test, y_test)



plt.figure(figsize = (15,10))   

plt.title('Different variation neighbors number')

plt.plot(neighbors, test_accuracy, label = 'Test Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Train Accuracy')

plt.legend()

plt.xlabel('Nº of Neighbors')

plt.ylabel('Neighbor accuracy')

plt.show()
k_range = range(1, 30)

scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)
print('Test set score: {:.2f}'.format(knn.score(X_test, y_test)))
classes = {0:'setosa',1:'versicolor',2:'virginica'}



X_new = [[3,4,5,2],

         [5,3,2,2],

         [4,1,2,1],

         [8,6,2,1],

         [3,2,1.5,2],

         [2,4,2,1],

         [4,4,2.4,1],

         [5,4,2,2],

         [6,1,2,3],

         [3,4,1.9,2],

         [2,4,2,1],

         [3,4,1.8,1],

         [3,3,2,2],

         [5,4,2,1]]

        





y_predict = knn.predict(X_new)



print('Test predictions:{}'.format(y_predict))

print(classes[y_predict[0]])

print(classes[y_predict[1]])

print(classes[y_predict[2]])

print(classes[y_predict[3]])

print(classes[y_predict[4]])

print(classes[y_predict[5]])

print(classes[y_predict[6]])

print(classes[y_predict[7]])

print(classes[y_predict[8]])

print(classes[y_predict[9]])

print(classes[y_predict[10]])

print(classes[y_predict[11]])

print(classes[y_predict[12]])

print(classes[y_predict[13]])
