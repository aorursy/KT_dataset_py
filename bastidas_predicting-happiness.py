import numpy as np

import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
happy_index = pd.read_csv("../input/2017.csv", sep=',', header=0, names=["country", "rank", "score", "high", "low", "gdp", "family", "health", "free", "gen", "trust","dystopia"])

happy_index.head(5)
corr_matrix = happy_index.corr()

sns.heatmap(corr_matrix, annot=True, cbar=True, cmap="RdYlGn")

plt.show()
cols = ["score","high", "low", "family", "gdp", "free", "health", "gen", "trust"]

happy_index_simple = happy_index[cols]

scaler = MinMaxScaler()

happy_index_simple[cols] = scaler.fit_transform(happy_index[cols])

corr_matrix = happy_index[["score", "family", "gdp", "free", "health", "gen", "trust"]].corr()

sns.set_context("poster",font_scale=1.0) 

sns.heatmap(corr_matrix, annot=True, cbar=True, cmap="GnBu")

plt.show()
metrics = ["family", "gdp", "free", "health", "gen", "trust"]

intercepts = {}

coefs = {}

f, axes  = plt.subplots(3, 2, sharey='row')

n = 0 

for metric in metrics:    

    X = happy_index_simple[metric]

    X = X.values.reshape(-1,1)

    y = happy_index_simple["score"]

    y = y.values.reshape(-1,1)

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)

    lr = LinearRegression(normalize=False, fit_intercept=True)

    lr.fit(X_train,y_train)

    intercepts[metric] = lr.intercept_[0]

    coefs[metric] = lr.coef_[0][0]

    predictions = lr.predict(X_test)

    axes[n%3][n%2].scatter(X_train,y_train, color = "grey")

    #axes[n%3][n%2].scatter(X_test,y_test, color = "black")

    axes[n%3][n%2].plot(X_train, lr.coef_[0][0]*X_train+ lr.intercept_[0], color="black", alpha=.7)

    axes[n%3][n%2].set_title(metric)

    n += 1

f.subplots_adjust(hspace=0.4)

plt.show()

print(intercepts)

print(coefs)
def plot_errors(X_test,y_test,predictions):

    f, axes  = plt.subplots(3, 2, sharey='row')

    n = 0 

    for metric in metrics:

    #axes[n%3][n%2].scatter(X_test[metric].values, y_test, color = "black")

        axes[n%3][n%2].scatter(X_test[metric].values, y_test-predictions, color = "grey", alpha=.7)

        axes[n%3][n%2].set_title(metric)

        n += 1

        f.subplots_adjust(hspace=0.4)

    plt.show()
X = happy_index_simple[metrics]

y = happy_index_simple["score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=111)
mlr = LinearRegression(normalize=False, fit_intercept=True)

mlr.fit(X_train, y_train)

print('Coeff:', mlr.coef_)

print('Intercept:', mlr.intercept_)

mlr_predictions = mlr.predict(X_test)

print('mae:', mean_absolute_error(y_test, mlr_predictions))

print('mse:', mean_squared_error(y_test, mlr_predictions))

print('rmse:', np.sqrt(mean_squared_error(y_test, mlr_predictions)))

print('score:', mlr.score(X_test, y_test))

plot_errors(X_test, y_test, mlr_predictions)
gbr = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls",

                                              max_features=0.9, min_samples_leaf=5,

                                              min_samples_split=6)

param_grid = [

    {'n_estimators': [96,128,512],

    'min_samples_leaf':[1,5],

    'alpha': [.85,.9, .95]}

    #'min_impurity_split': [1e-08,1e-06,1e-05],

    #'max_features': [.5,0.9,1],

    #'min_samples_split': [2,6]}

    ]

gbr_grid = GridSearchCV(gbr, cv=2, n_jobs=2, param_grid=param_grid, scoring="neg_mean_squared_error")

gbr_grid.fit(X,y)

gbr_cv = gbr_grid.best_estimator_

#print(gbr_cv.get_params)

gbr_cv.fit(X_train,y_train)

gbr_predictions = gbr_cv.predict(X_test)

print('mae:', mean_absolute_error(y_test, gbr_predictions))

print('mse:', mean_squared_error(y_test, gbr_predictions))

print('rmse:', np.sqrt(mean_squared_error(y_test, gbr_predictions)))

print('score:', gbr_cv.score(X_test, y_test))

plot_errors(X_test, y_test, gbr_predictions)

X = happy_index_simple[metrics] 

y = happy_index_simple["score"]

xrange_vals = range(len(X))

xrange_vals = np.asarray(xrange_vals).reshape(-1,1)



#multiple linear regression

mlr_error = - y + mlr.predict(X)

plt.plot(xrange_vals, mlr_error, color="blue", alpha=.2, label='multiple linear regression')

mlr_meta = LinearRegression(normalize=False, fit_intercept=True)

mlr_meta.fit(xrange_vals, mlr_error)

plt.plot(xrange_vals,mlr_meta.predict(xrange_vals), color="blue")



#gradient boosted regression

gbr_error = - y + gbr_cv.predict(X)

plt.plot(xrange_vals, gbr_error, color="red", alpha=.2, label='gradient boosted regression')

gbr_meta = LinearRegression(normalize=False, fit_intercept=True)

gbr_meta.fit(xrange_vals, gbr_error)

plt.plot(xrange_vals, gbr_meta.predict(xrange_vals), color="red")



plt.xlabel("country rank")

plt.ylabel("error")

plt.legend()

plt.show()
gbr_error = - y + gbr_cv.predict(X)

plt.plot(xrange_vals, gbr_error, color="red", alpha=.5, label='gradient boosted regression')

gbr_meta = LinearRegression(normalize=False, fit_intercept=True)

gbr_meta.fit(xrange_vals, gbr_error)

plt.plot(xrange_vals, gbr_meta.predict(xrange_vals), color="red")

plt.plot(xrange_vals, happy_index_simple["score"]-happy_index_simple["high"], color="black", alpha=.5, label="error")

plt.plot(xrange_vals, happy_index_simple["score"]-happy_index_simple["low"], color="black", alpha=.5)

plt.xlabel("country rank")

plt.ylabel("error")

plt.legend()

plt.show()



happy_index_simple["delta"] = gbr_error

happy_index_simple["pred"] = gbr_cv.predict(X)

print(happy_index_simple.head(5))