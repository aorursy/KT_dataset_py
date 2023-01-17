import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline
data = pd.read_csv("../input/housing.csv")

data.head()
data.shape
data.info()
# Pearson correlation

plt.subplots(figsize=(15, 9))

cor = data.corr()

sns.heatmap(cor, annot=True, linewidths=.5)

plt.show()
# taking two variables

data = data.drop(["housing_median_age","households","total_bedrooms","longitude","latitude","total_rooms","population","ocean_proximity"], axis=1)

data.head()
X = data.drop("median_house_value", axis=1)

y = data["median_house_value"]
plt.scatter(X, y, alpha=0.5)

plt.title('Scatter plot')

plt.xlabel('median_income')

plt.ylabel('median_house_value')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



# Model initialization

regression_model = LinearRegression(normalize = True)



# Fit the data(train the model)

regression_model.fit(X_train, y_train)
# Predict

y_predicted = regression_model.predict(X_test)



# model evaluation

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

r2 = r2_score(y_test, y_predicted)
# printing values

print('Slope:' ,regression_model.coef_)

print('Intercept:', regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)
# data points

plt.scatter(X_train, y_train, s=10)

plt.xlabel('median_income')

plt.ylabel('median_house_value')



# predicted values

plt.plot(X_test, y_predicted, color='r')

plt.show()
residual = y_test - y_predicted

sns.residplot(residual,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plt.show()
tf = np.sqrt(X_train) 

tf1 = np.sqrt(X_test)



plt.scatter(tf, y_train)

plt.show()
regression_model.fit(tf, y_train)

# Predict

y_predicted = regression_model.predict(tf1)



# model evaluation

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

r2 = r2_score(y_test, y_predicted)



# printing values

print('Slope:' ,regression_model.coef_)

print('Intercept:', regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)
res = y_test - y_predicted

sns.residplot(res,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plt.show()
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X_train)

pol_reg = LinearRegression(normalize = True)

pol_reg.fit(X_poly, y_train)
def viz_polymonial():

    plt.scatter(X_train, y_train, color="red")

    plt.plot(X_train, pol_reg.predict(poly_reg.fit_transform(X_train)))

    plt.xlabel('median_income')

    plt.ylabel('median_house_value')

    plt.show()

    return

viz_polymonial()
# Predict

X_p = poly_reg.fit_transform(X_test)

y_predicted = pol_reg.predict(X_p)



# model evaluation

rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

r2 = r2_score(y_test, y_predicted)



# printing values

print('Slope:' ,regression_model.coef_)

print('Intercept:', regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)
res = y_test - y_predicted

sns.residplot(res,y_predicted, lowess=True,scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plt.show()