import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split

data = pd.read_csv("../input/random-linear-regression/train.csv")

test = pd.read_csv("../input/random-linear-regression/test.csv")
data.head()
data = data.dropna()
test = test.dropna()
data.info()
X_train = data[["x"]]
y_train = data[["y"]]
X_test = test[["x"]]
y_test = test[["y"]]
lm = LinearRegression()
lm.fit(X_train, y_train)
print("Coefficient of X: "+ str(lm.coef_))
print("Intercept: "+ str(lm.intercept_))
cdf = pd.DataFrame(lm.coef_, X_train.columns, columns=["Coeff"])
cdf
predictions = lm.predict(X_test)
# Scatter Plot
plt.scatter(y_test, predictions)
# Histogram for the Residuals
yhat = y_test-predictions

sns.distplot(yhat)

plt.title("Histogram of Residuals")
print("Mean Absolute Error: "+ str(metrics.mean_absolute_error(y_test, predictions)))

print("Mean Squared Error: " + str(metrics.mean_squared_error(y_test,predictions)))

print("Root Mean Squared Error: " + str(np.sqrt(metrics.mean_squared_error(y_test, predictions))))