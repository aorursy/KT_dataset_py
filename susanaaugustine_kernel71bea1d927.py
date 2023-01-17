from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data

Y = iris.target

iris.target_names

iris.data.shape

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=4)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

k_range = range(1,26)

scores = {}

scores_list = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train,Y_train)

    Y_pred = knn.predict(X_test)

    scores[k] = metrics.accuracy_score(Y_test,Y_pred)

    scores_list.append(metrics.accuracy_score(Y_test,Y_pred))

%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(k_range,scores_list)

plt.xlabel('value of K for KNN')

plt.ylabel('Testing Accuracy')

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X,Y)

classes = {0:'setosa', 1:'versicolor',2: 'virginica'}

X_new = [[3,4,2,2],[5,2,2,2]]

Y_predict = knn.predict(X_new)

print(classes[Y_predict[0]])

print(classes[Y_predict[1]])