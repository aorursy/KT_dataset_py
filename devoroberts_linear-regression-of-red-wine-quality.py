import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
data = pd.read_csv('../input/winequality-red.csv')
data.head()
data.shape
data[data.isnull().any(axis=1)]
data.shape
data.columns
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']
target = ['quality']
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=324)
regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred_quality = regressor.predict(X_test)
pred_quality
y_test.describe()
RMSE = mean_squared_error(y_true=y_test, y_pred=pred_quality)
print(RMSE)
regressor = DecisionTreeRegressor(max_depth=30)
regressor.fit(X_train, y_train)
pred_quality = regressor.predict(X_test)
pred_quality
y_test.describe()
RMSE = mean_squared_error(y_true=y_test, y_pred=pred_quality)
print(RMSE)