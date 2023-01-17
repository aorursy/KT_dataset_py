import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import plotly.graph_objs as go



init_notebook_mode(connected=True)

import plotly.offline as offline
print("""











EXERCISE 01 : 

Use the dataset IRIS of scikit-learn to illustrate how to use the KNN algorithm with

splitting data.













""")
import os

os.listdir("../input/iris-dataset")
iris = pd.read_csv("../input/iris-dataset/iris.csv")

iris.fillna(method = "bfill", inplace = True)

iris.head()
iris.info()
iris.describe()
sns.pairplot(data = iris, hue='species')
"""Model Preparation"""
from sklearn.model_selection import train_test_split
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

y = iris["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))

print("X_test shape: {}\ny_test shape: {}".format(X_test.shape, y_test.shape))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
df = pd.concat([X_test, y_test, pd.Series(y_pred, name='Predicted', index=X_test.index)], ignore_index=False, axis=1)
df.head()
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
print("""









EXERCISE 02:

Use the dataset IRIS of scikit-learn to illustrate how to use pipeline, scaling, grid

search and the KNN algorithm.











""")
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection  import GridSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# normilaizer data

# classification

pipeline = Pipeline([

    ('normalizer', StandardScaler()), 

    ('clf', LogisticRegression()) 

])
scores = cross_validate(pipeline, X_train, y_train)

scores
scores["test_score"].mean()
knn=KNeighborsClassifier(n_neighbors=5)
k_range=range(1,31)

param_grid=dict(n_neighbors=k_range)

grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X = X_train,y = y_train)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
weight_options=['uniform','distance']

param_grid=dict(n_neighbors=k_range,weights=weight_options)

print (param_grid)
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(X_train,y_train)
print (grid.best_score_)

print (grid.best_params_)
knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')

knn.fit(X_train,y_train)

pre = knn.predict(X_test)
knn.score(X_test, y_test)
X_test["species"] = y_test

X_test["prediction"] = pre

X_test.head()
print("""











EXERCISE 03:

Use the dataset IRIS of scikit-learn to illustrate how to use the logistic regression

algorithm with splitting data.













""")
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

y = iris["species"]
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)

logreg.score(X_test, y_test)
X_test["species"] = y_test

X_test["prediction"] = y_pred

X_test.head()
print("""





EXERCISE 05:

Illustrate how to use the linear regression algorithm with splitting data on the Boston

housing dataset.





""")
bs = pd.read_csv("../input/boston-house/boston.csv")

bs.head()
plt.figure(figsize=(20,10)) 

sns.heatmap(bs.corr(),annot=True,cmap='cubehelix_r') 

plt.show()
from sklearn.linear_model import LinearRegression
X = bs["RM"].values.reshape(-1, 1)

y = bs["MEDV"].values.reshape(-1, 1)
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)
lr = LinearRegression()
lr.fit(X_train, y_train)
plt.figure(figsize = (20, 10))

plt.scatter(X_train, y_train)

plt.plot(X_train, lr.predict(X_train), color =  "red")

plt.xlabel("RM")

plt.ylabel("MEDV")

plt.title("Trainning Set : RM vs MEDV")
plt.figure(figsize = (20, 10))

plt.scatter(X_test, y_test)

plt.plot(X_train, lr.predict(X_train), color =  "red")

plt.xlabel("RM")

plt.ylabel("MEDV")

plt.title("Test Set : RM vs MEDV")
X = bs["LSTAT"].values.reshape(-1, 1)

y = bs["MEDV"].values.reshape(-1, 1)
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3,random_state=101)
lr = LinearRegression()
lr.fit(X_train, y_train)
plt.figure(figsize = (20, 10))

plt.scatter(X_train, y_train)

plt.plot(X_train, lr.predict(X_train), color =  "red")

plt.xlabel("LSTAT")

plt.ylabel("MEDV")

plt.title("Trainning Set : LSTAT vs MEDV")
plt.figure(figsize = (20, 10))

plt.scatter(X_test, y_test)

plt.plot(X_train, lr.predict(X_train), color =  "red")

plt.xlabel("LSTAT")

plt.ylabel("MEDV")

plt.title("Test Set : LSTAT vs MEDV")