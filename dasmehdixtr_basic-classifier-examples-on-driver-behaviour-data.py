import pandas as pd
data = pd.read_csv("/kaggle/input/carla-driver-behaviour-dataset/full_data_carla.csv",index_col=0)

data.head()
data.describe()
data.info()
data['class'].unique()
x = data.drop(["class"],axis=1)

y = data["class"].values

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

X_train.head()
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train,y_train)

nn = MLPClassifier(solver='lbfgs')

nn.fit(X_train,y_train)

nb = GaussianNB()

nb.fit(X_train,y_train)

knn = KNeighborsClassifier(n_neighbors = 3) #n_neighbors = k

knn.fit(X_train,y_train)

svm = SVC(random_state = 1)

svm.fit(X_train,y_train)

print("SVM accuracy is :",svm.score(X_test,y_test))

print("k={} nn score={}".format(3,knn.score(X_test,y_test)))

print('Naive Bayes Accuracy= :', nb.score(X_test,y_test))

print('MLP Accuracy= ',nn.score(X_test,y_test))

print('SGD Accuracy=: ', sgd.score(X_test,y_test))
from sklearn.model_selection import cross_val_score

import numpy as np

accuracy = cross_val_score(estimator = knn, X = X_train, y =y_train, cv = 8)

print("avg acc: ",np.mean(accuracy))

print("acg std: ",np.std(accuracy))