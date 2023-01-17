import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
np.random.seed(0)



n_samples = 400



def true_fun(X_1, X_2 , X_3):

    

    y_temp_1 = np.cos(1.5 * np.pi * X_1)

    y_temp_2 = y_temp_1 * X_2 * X_2

    y_temp_3 = X_3 * 0.5

    

    result = 2 * y_temp_1 + y_temp_2 + y_temp_3

    return result



X_1 = np.sort(np.random.rand(n_samples))

X_2 = np.sort(np.random.rand(n_samples)) +  np.random.randn(n_samples) * 0.2



data = pd.DataFrame(X_1, columns= {"X_1"})

data["X_2"] = X_2

data["X_3"] = 1



data["X_3"].loc[data["X_2"] >0.4] = 0.5

y = true_fun(data["X_1"], data["X_2"], data["X_3"]) + np.random.randn(n_samples) * 0.2



data["y"] = y

sns.pairplot(data, kind="reg" , diag_kind="kde")
X = data.drop("y" , axis = 1)

y = data["y"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

print (lm.score(X_train,y_train))

print (lm.score(X_test,y_test))
data["X_4"] = data["X_1"] * data["X_1"]
X = data.drop("y" , axis = 1)

y = data["y"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict( X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

print (lm.score(X_train,y_train))

print (lm.score(X_test,y_test))
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

polynomial_features = PolynomialFeatures(degree=3,include_bias=False)

linear_regression = LinearRegression()

pipeline = Pipeline([("polynomial_features", polynomial_features),

                         ("linear_regression", linear_regression)])

pipeline.fit(X_train,y_train)

predictions = pipeline.predict( X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')



print (pipeline.score(X_train,y_train))

print (pipeline.score(X_test,y_test))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=100, max_features = 4)

rf.fit(X_train,y_train)

print(regr.feature_importances_)

predictions = rf.predict( X_test)

plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

print (rf.score(X_train,y_train))

print (rf.score(X_test,y_test))