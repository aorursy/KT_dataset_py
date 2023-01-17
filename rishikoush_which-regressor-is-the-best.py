import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/Admission_Predict.csv')
data.head()
data.info()
data.isnull().sum()
y = data['Chance of Admit ']

X = data.drop(['Chance of Admit '],axis=1)
y.shape
from sklearn.metrics import mean_squared_error
def run_model(X_train, y_train, X_test,y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = ((mean_squared_error(y_test,y_pred)) ** (0.5))

    return rmse
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

print('RMSE for linear regression:' ,run_model(X_train, y_train, X_test,y_test))
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

print('RMSE for Decision Tree Regressor:' ,run_model(X_train, y_train, X_test,y_test))
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.5)

print('RMSE for Ridge Regressor:' ,run_model(X_train, y_train, X_test,y_test))
from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1)

print('RMSE for Lasso Regressor:' ,run_model(X_train, y_train, X_test,y_test))
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.5)

print('RMSE for Elastic Net Regressor:' ,run_model(X_train, y_train, X_test,y_test))
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(max_depth=7)

print('RMSE for Random Forest Regressor:' ,run_model(X_train, y_train, X_test,y_test))