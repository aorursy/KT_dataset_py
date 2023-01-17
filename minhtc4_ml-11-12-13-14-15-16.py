import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import plotly as py

from plotly.offline import iplot, plot, init_notebook_mode, download_plotlyjs

import plotly.graph_objs as go



init_notebook_mode(connected=True)

import plotly.offline as offline
db = pd.read_csv("../input/diabetes-dataset/diabetes.csv")

db.head()
X = db[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']]

y = db["Outcome"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import cross_validate

from sklearn.model_selection  import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression
pipeline = Pipeline([

    ('normalizer', StandardScaler()), 

    ('knn', KNeighborsClassifier()) 

])

scores = cross_validate(pipeline, X_train, y_train)

scores["test_score"].mean()

knn=KNeighborsClassifier(n_neighbors=5)

k_range=range(1,31)

param_grid=dict(n_neighbors=k_range)

grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')



grid.fit(X = X_train,y = y_train)

print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
knn=KNeighborsClassifier(n_neighbors= 12,weights='uniform')

knn.fit(X_train,y_train)

pre = knn.predict(X_test)

knn.score(X_test, y_test)

print("""





EXERCISE 12

Illustrate how to use the linear regression algorithm with splitting data for

gapminder.csv





""")
gm = pd.read_csv("../input/gapminder1/gapminder.csv")

gm.head()
X = gm[['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP','BMI_female']]

y = gm["life"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
leg = LinearRegression()

leg.fit(X_train, y_train)
leg.predict(X_test)

leg.score(X_test, y_test)
X = gm[['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP','BMI_female']]

y = gm["child_mortality"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
leg = LinearRegression()

leg.fit(X_train, y_train)
leg.predict(X_test)

leg.score(X_test, y_test)
print("""





EXERCISE 13

Illustrate how to use the K-Means algorithm with splitting data for gapminder.csv.





""")
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

pca.fit(X)
X
print("""





EXERCISE 16

Write a program to compare different models to predict life and child mortality on

gapminder.csv.





""")
gm.columns
X_l = gm[['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP','BMI_female']]

y_l = gm["life"]

y_c = gm["child_mortality"]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_l, y_l)

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_l, y_c)



print("""



life



""")
"""Linear Regression"""
reg = LinearRegression()
reg.fit(X_train, y_train)
lf_lin = reg.score(X_test, y_test)

lf_lin