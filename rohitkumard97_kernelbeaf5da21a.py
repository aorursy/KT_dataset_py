import pandas as pd

import numpy as np
data = pd.read_csv("../input/predict-admission/Admission.csv")

data1 = data.drop(['Serial No.'],axis=1)

data1.shape
data1.head()
x = data1.iloc[:, :-1].values

y = data1.iloc[:, 7].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
accuracy = regressor.score(x_test,y_test)

accuracy*100
y_pred = regressor.predict(x_test)
y_pred
y_test
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
accuracy = regressor.score(x_test,y_test)

accuracy*100
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

accuracy = regressor.score(x_test,y_test)

accuracy*100
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

accuracy = regressor.score(x_test,y_test)

accuracy*100