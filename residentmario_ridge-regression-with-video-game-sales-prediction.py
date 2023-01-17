import pandas as pd

import numpy as np

sales = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

sales.head()
import seaborn as sns

s = sales[(sales.NA_Sales.notnull() & sales.JP_Sales.notnull())]

s = sales[((sales.NA_Sales > 0) & (sales.JP_Sales > 0))]

s = s.sample(100, random_state=0)

s = s.loc[s.NA_Sales.rank().sort_values().index]

sns.jointplot(s.NA_Sales.rank(), s.JP_Sales)
NA_sales_ranks = s.NA_Sales.rank().values[:, np.newaxis]

JP_sales = s.JP_Sales.values[:, np.newaxis]
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



def predict(degree):

    poly = PolynomialFeatures(degree=degree)

    NA_sales_ranks_poly = poly.fit_transform(NA_sales_ranks)

    clf = LinearRegression()

    clf.fit(NA_sales_ranks_poly, JP_sales)

    JP_sale_predictions = clf.predict(NA_sales_ranks_poly)

    result = np.dstack((JP_sales.flatten(), JP_sale_predictions.flatten())).reshape((100, 2))

    return result
result = predict(1)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("Regression with Degree 1")
result = predict(3)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("Regression with Degree 3")
result = predict(10)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("Regression with Degree 10")
def get_model(degree):

    poly = PolynomialFeatures(degree=degree)

    NA_sales_ranks_poly = poly.fit_transform(NA_sales_ranks)

    clf = LinearRegression()

    clf.fit(NA_sales_ranks_poly, JP_sales)

    return clf



m1_coef = get_model(1).coef_

m3_coef = get_model(3).coef_

m10_coef = get_model(10).coef_
m1_coef
m3_coef
m10_coef
from sklearn.linear_model import Ridge



def get_ridge_model(degree, alpha):

    poly = PolynomialFeatures(degree=degree)

    NA_sales_ranks_poly = poly.fit_transform(NA_sales_ranks)

    clf = Ridge(alpha=alpha)

    clf.fit(NA_sales_ranks_poly, JP_sales)

    return clf



def ridge_predict(degree, alpha):

    poly = PolynomialFeatures(degree=degree)

    NA_sales_ranks_poly = poly.fit_transform(NA_sales_ranks)

    clf = get_ridge_model(degree, alpha)

    JP_sale_predictions = clf.predict(NA_sales_ranks_poly)

    result = np.dstack((JP_sales.flatten(), JP_sale_predictions.flatten())).reshape((100, 2))

    return result
result = ridge_predict(10, 10)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("degree=10, alpha=$10^1$")
result = ridge_predict(10, 10**25)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("degree=10, alpha=$10^{25}$")
result = ridge_predict(10, 10**40)

plt.plot(range(len(result[:, 1])), result[:, 1])

plt.scatter(range(len(result[:, 0])), result[:, 0], color='black')

plt.gca().set_title("degree=10, alpha=$10^{40}$")
get_ridge_model(10, 10**40).coef_