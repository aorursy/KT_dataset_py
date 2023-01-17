import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor

data = pd.read_csv('../input/movie_metadata.csv') 

data.director_name = data.director_name.factorize()[0]

data.actor_1_name = data.actor_2_name.factorize()[0]

data.actor_2_name = data.actor_2_name.factorize()[0]

data.actor_3_name = data.actor_3_name.factorize()[0]

data.color = data.color.factorize()[0]

data.language = data.language.factorize()[0]

data.country = data.country.factorize()[0]

data.content_rating = data.content_rating.factorize()[0]

genres = []

[[genres.append(j) for j in i.split("|")] for i in Counter(data.genres).keys()]

genres = list(set(genres))

for i in genres:

    temp = []

    for j in data.genres:

        if i in j.split("|"):

            temp.append(1)

        else:

            temp.append(0)

    data.insert(2, i, temp)

data = data.drop('plot_keywords', axis=1)

print (data.describe())

data = data.drop('movie_imdb_link', axis=1)

data = data.drop('movie_title', axis=1)

data = data.drop('genres', axis=1)

data = data.dropna()

y = data.imdb_score

data = data.drop('imdb_score', axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25)

model = RandomForestRegressor(n_estimators = 50, n_jobs = -1, max_features = 15, max_depth = 15)

model.fit(X_train,y_train)

print (model.score(X_train,y_train))

print (model.score(X_test, y_test))

y_predictions = model.predict(X_test)

#print(y_predictions)

#print(y_test.values)

error = []

for i in range(len(y_test.values)):

    if abs(y_predictions[i] - y_test.values[i]) <= 0.05 * y_predictions[i]:

        error.append(True)

    else:

        error.append(False)

print('Error = ', Counter(error))

num = [i for i in range(len(y_test.values))]

plt.scatter(num,y_test.values, color = "r")

#plt.scatter(num,y_predictions)

plt.show()