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











EXERCISE 06

Illustrate how to use the K-Means algorithm with splitting data on the Boston housing

dataset.













""")
bs = pd.read_csv("../input/boston-house/boston.csv")

bs.head()
X = bs[['RM','LSTAT']]

y = bs["MEDV"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 101, test_size = 0.3)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X_train)

labels = kmeans.predict(X_train)

centroids = kmeans.cluster_centers_

centroids
centroids[0][0]
temp = X_train

temp["mean"] = labels

temp["price"] = y_train

temp1 = temp[temp["mean"] == 0]

temp2 = temp[temp["mean"] == 1]

temp3 = temp[temp["mean"] == 2]
mn1 = temp1["price"].mean()

mn2 = temp2["price"].mean()

mn3 = temp3["price"].mean()

trace1 = go.Scatter3d(

    x = temp1.RM,

    y = temp1.LSTAT,

    z = temp1.price,

    mode = "markers",

    marker=dict(

        color= "red",

        opacity=0.8,

        size = 8

    )

)

trace2 = go.Scatter3d(

    x = temp2.RM,

    y = temp2.LSTAT,

    z = temp2.price,

    mode = "markers",

    marker=dict(

        color= "blue",

        opacity=0.8,

        size = 8

    )

)

trace3 = go.Scatter3d(

    x = temp3.RM,

    y = temp3.LSTAT,

    z = temp3.price,

    mode = "markers",

    marker=dict(

        color= "yellow",

        opacity=0.8,

        size = 8

    )

)

trace4 = go.Scatter3d(

    x = [centroids[0][0], centroids[1][0], centroids[2][0]],

    y = [centroids[0][1], centroids[1][1], centroids[2][1]],

    z = [mn1, mn2, mn3],

    mode = "markers",

    marker = dict(

        color = "black",

        size = 16

    )

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    title  = "3DX"

)

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)
kk = kmeans.predict(X_test)
test_after = X_test

test_after["mean"] = kk

test_after["price"] = y_test

tp1 = test_after[test_after["mean"] == 0]

tp2 = test_after[test_after["mean"] == 1]

tp3 = test_after[test_after["mean"] == 2]
trace1 = go.Scatter3d(

    x = tp1.RM,

    y = tp1.LSTAT,

    z = tp1.price,

    mode = "markers",

    marker=dict(

        color= "red",

        opacity=0.8,

        size = 8

    )

)

trace2 = go.Scatter3d(

    x = tp2.RM,

    y = tp2.LSTAT,

    z = tp2.price,

    mode = "markers",

    marker=dict(

        color= "blue",

        opacity=0.8,

        size = 8

    )

)

trace3 = go.Scatter3d(

    x = tp3.RM,

    y = tp3.LSTAT,

    z = tp3.price,

    mode = "markers",

    marker=dict(

        color= "yellow",

        opacity=0.8,

        size = 8

    )

)

trace4 = go.Scatter3d(

    x = [centroids[0][0], centroids[1][0], centroids[2][0]],

    y = [centroids[0][1], centroids[1][1], centroids[2][1]],

    z = [mn1, mn2, mn3],

    mode = "markers",

    marker = dict(

        color = "black",

        size = 16

    )

)

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    title  = "3DX"

)

fig = go.Figure(data = data, layout = layout)

offline.iplot(fig)
print("""











EXERCISE 07 :

Illustrate how to use the Hierarchical Clustering algorithm with splitting data on the

Boston housing dataset.













""")
print("""











EXERCISE 08

Illustrate how to use the KNN algorithm with splitting data for diabetes.csv













""")
db = pd.read_csv("../input/diabetes-dataset/diabetes.csv")
db.head()
plt.figure(figsize = (20, 10))

sns.heatmap(db.corr(), annot = True)
X = db[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = db["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
df = pd.concat([X_test, y_test, pd.Series(y_pred, name='l', index=X_test.index)], ignore_index=False, axis=1)
knn.score(X_test, y_test)
print("""











EXERCISE 09

Illustrate how to use pipeline, scaling, grid search and the KNN algorithm for

diabetes.csv.













""")
db.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import cross_validate

from sklearn.model_selection  import GridSearchCV
pipeline = Pipeline([

    ('normalizer', StandardScaler()), 

    ('knn', KNeighborsClassifier()) 

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
knn=KNeighborsClassifier(n_neighbors= 27,weights='uniform')

knn.fit(X_train,y_train)

pre = knn.predict(X_test)
knn.score(X_test, y_test)
print("""











EXERCISE 10

Illustrate how to use the logistic regression algorithm with splitting data for

diabetes.csv











""")
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
logreg = LogisticRegression()

logreg.fit(X_train,y_train)

y_pred= logreg.predict(X_test)
# logreg.score(X_test, y_test)\

accuracy_score(y_pred, y_test)
