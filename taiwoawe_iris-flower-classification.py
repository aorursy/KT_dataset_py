import pandas as pd    # data structure library

import numpy as np         # for scientific computing

import matplotlib.pyplot as plt      # for ploting in python

from sklearn.model_selection import train_test_split   # to split our dataset into two

from sklearn.linear_model import LogisticRegression  

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df.head()
data = df.copy()

encode = LabelEncoder()

data['species'] = encode.fit_transform(data['species'])
data.head()
x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_length']]  # features

x.head()
y = data['species']   # the label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
reg = LogisticRegression() # perform a logistic regression 

reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

y_pred
accuracy_score(y_test, y_pred)
neigh = KNeighborsClassifier(n_neighbors = 4) #KNN algorithm

neigh.fit(x_train,y_train)
yhat = neigh.predict(x_test)
yhat[0:10]
accuracy_score(y_test, yhat)
clf = svm.SVC(kernel='rbf')   # support vector machine

clf.fit(x_train, y_train)
predict_y = clf.predict(x_test)

predict_y[0:10]
accuracy_score(y_test, predict_y)
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) # decision tree classifier

Tree.fit(x_train,y_train)
prediction = Tree.predict(x_test)

prediction[0:10]
accuracy_score(y_test, prediction)