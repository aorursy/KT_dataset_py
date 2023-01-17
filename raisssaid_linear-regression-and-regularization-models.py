import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
%matplotlib inline
data = pd.read_csv("../input/advertising-data/Advertising.csv")
data.head()
data.drop(['Unnamed: 0'], axis=1, inplace=True)
X = data.drop(['Sales', 'Newspaper'], axis=1)
Y = data['Sales'].values.reshape(-1,1)
X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=42
    )
def print_evaluate(y_test, predicted):  
    mae = metrics.mean_absolute_error(y_test, predicted)
    mse = metrics.mean_squared_error(y_test, predicted)
    r2_square = metrics.r2_score(y_test, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2 Square', r2_square)
def cross_val(model, X = X, Y = Y, cv=10):
    MSE = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=cv)
    return print(-np.mean(MSE))
# Linear Regression
linear_reg = LinearRegression()
# Ridge Regression
ridge = Ridge()
# Lasso
lasso = Lasso(tol=1e4)
# Elastic Net
elastic_net = ElasticNet()
linear_reg.fit(X_train, Y_train)
linear_reg_pred = linear_reg.predict(X_test)
print_evaluate(Y_test, linear_reg_pred)
results = pd.DataFrame(
    data = [
            ["Linear Regression",
             metrics.mean_squared_error(Y_test, linear_reg_pred),
             metrics.r2_score(Y_test, linear_reg_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
parameters = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
ridgeCV = GridSearchCV(
    ridge, parameters, scoring='neg_mean_squared_error', cv=5
    )
ridgeCV.fit(X_train, Y_train)
ridgeCV_pred = ridgeCV.predict(X_test)
print_evaluate(Y_test, ridgeCV_pred)
ridge_result = pd.DataFrame(
    data = [
            ["Ridge Regression",
             metrics.mean_squared_error(Y_test, ridgeCV_pred),
             metrics.r2_score(Y_test, ridgeCV_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(ridge_result, ignore_index=True)
parameters =  {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lassoCV = GridSearchCV(
    lasso, parameters, scoring='neg_mean_squared_error', cv=5,
    )
lassoCV.fit(X_train, Y_train)
lassoCV_pred = lassoCV.predict(X_test)
print_evaluate(Y_test, lassoCV_pred)
lasso_result = pd.DataFrame(
    data = [
            ["Lasso",
             metrics.mean_squared_error(Y_test, lassoCV_pred),
             metrics.r2_score(Y_test, lassoCV_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(lasso_result, ignore_index=True)
elastic_net_default = ElasticNet()
elastic_net_default.fit(X_train, Y_train)
elastic_net_default_pred = elastic_net_default.predict(X_test)
print_evaluate(Y_test, elastic_net_default_pred)
elastic_net_default_result = pd.DataFrame(
    data = [
            ["Elastic Net (alpha=1)",
             metrics.mean_squared_error(Y_test, elastic_net_default_pred),
             metrics.r2_score(Y_test, elastic_net_default_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(elastic_net_default_result, ignore_index=True)
elastic_net = ElasticNet(alpha = 10.5)
elastic_net.fit(X_train, Y_train)
elastic_net_pred = elastic_net.predict(X_test)
print_evaluate(Y_test, elastic_net_pred)
elastic_net_result = pd.DataFrame(
    data = [
            ["Elastic Net (alpha=10.5)",
             metrics.mean_squared_error(Y_test, elastic_net_pred),
             metrics.r2_score(Y_test, elastic_net_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(elastic_net_result, ignore_index=True)
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
ransac = RANSACRegressor()
ransac.fit(X_train, Y_train)
ransac_pred = ransac.predict(X_test)
print_evaluate(Y_test, ransac_pred)
ransac_result = pd.DataFrame(
    data = [
            ["Robust RANSAC Regressor",
             metrics.mean_squared_error(Y_test, ransac_pred),
             metrics.r2_score(Y_test, ransac_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(ransac_result, ignore_index=True)
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(
    X_poly, Y, test_size=0.25, random_state=42
    )

lin_reg_poly = LinearRegression()

lin_reg_poly.fit(X_poly_train,y_poly_train)
poly_pred = lin_reg_poly.predict(poly_reg.transform(X_test))
print_evaluate(Y_test, poly_pred)
poly_result = pd.DataFrame(
    data = [
            ["Linear Regression using Polynomial Features",
             metrics.mean_squared_error(Y_test, poly_pred),
             metrics.r2_score(Y_test, poly_pred)
             ]
            ],
            columns=['Model', 'Mean Squared Error', 'R2 Square'])
results = results.append(poly_result, ignore_index=True)
results.sort_values(by=['Mean Squared Error', 'R2 Square'])