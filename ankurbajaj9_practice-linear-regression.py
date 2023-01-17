import sklearn

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
print(sklearn.__version__)
advertising_data = pd.read_csv('../input/advertising.csv/Advertising.csv', index_col=0)

advertising_data.head()
advertising_data.shape
advertising_data.describe()
plt.figure(figsize=(8, 8))

plt.scatter(advertising_data['newspaper'], advertising_data['sales'], c='y')

plt.show()
plt.figure(figsize=(8, 8))

plt.scatter(advertising_data['radio'], advertising_data['sales'], c='y')

plt.show()
plt.figure(figsize=(8, 8))

plt.scatter(advertising_data['TV'], advertising_data['sales'], c='y')

plt.show()
advertising_data_correlation = advertising_data.corr()

advertising_data_correlation
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 8))

sns.heatmap(advertising_data_correlation, annot=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = advertising_data['TV'].values.reshape(-1, 1)

Y = advertising_data['sales'].values.reshape(-1, 1)
X.shape, Y.shape
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
import statsmodels.api as sm

x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()

print(fit_model.summary())
linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)

linear_reg
print("Training_score : " , linear_reg.score(x_train, y_train))
y_pred = linear_reg.predict(x_test)
from sklearn.metrics import r2_score

print("Testing_score : ", r2_score(y_test, y_pred))
def adjusted_r2(r_square, labels, features):
    
    adj_r_square = 1 - ((1 - r_square) * (len(labels) - 1)) / (len(labels) - features.shape[1] - 1)
    
    return adj_r_square
print("Adjusted_r2_score : ", adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))
plt.figure(figsize=(8, 8))

plt.scatter(x_test,
            y_test,
            c='black')

plt.plot(x_test,
         y_pred,
         c='blue',
         linewidth=2)

plt.xlabel("Money spent on TV ads ($)")
plt.ylabel("Sales ($)")

plt.show()
X = advertising_data.drop('sales', axis=1)

Y = advertising_data['sales']
X.head(5)
Y.head()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)

fit_model = stats_model.fit()

print(fit_model.summary())
linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)

linear_reg
print("Training_score : " , linear_reg.score(x_train, y_train))
predictors = x_train.columns

coef = pd.Series(linear_reg.coef_, predictors).sort_values()

print(coef)
y_pred = linear_reg.predict(x_test)
print("Testing_score : ", r2_score(y_test, y_pred))
print("Adjusted_r2_score : ", adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))
plt.figure(figsize = (15, 8))

plt.plot(y_pred, label='Predicted')
plt.plot(y_test.values, label='Actual')

plt.ylabel("Sales ($)")
plt.legend()
plt.show()







































