# Import Pandas Library, used for data manipulation

# Import matplotlib, used to plot our data

# Import numpy for linear algebra operations

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

# Import our WeatherDataP.csv and store it in the variable weather_data_p

weather_data_p = pd.read_csv("../input/weather-data-p/WeatherDataP.csv")

 

# Display the data in the notebook

weather_data_p
# Set our input x to Pressure, use [[]] to convert to 2D array suitable for model input

X = weather_data_p[["Pressure (millibars)"]]

y = weather_data_p.Humidity



# Produce a scatter graph of Humidity against Pressure

plt.scatter(X, y, c = "black")

plt.xlabel("Pressure (millibars)")

plt.ylabel("Humidity")
# Import the function "PolynomialFeatures" from sklearn, to preprocess our data

# Import LinearRegression model from sklearn

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression



# Set PolynomialFeatures to degree 2 and store in the variable pre_process

# Degree 2 preprocesses x to 1, x and x^2

# Degree 3 preprocesses x to 1, x, x^2 and x^3

# and so on..

 

pre_process = PolynomialFeatures(degree=2)

# Transform our x input to 1, x and x^2

X_poly = pre_process.fit_transform(X)

# Show the transformation on the notebook

X_poly
pr_model = LinearRegression()



# Fit our preprocessed data to the polynomial regression model

pr_model.fit(X_poly, y)



# Store our predicted Humidity values in the variable y_new

y_pred = pr_model.predict(X_poly)



# Plot our model on our data

plt.scatter(X, y, c = "black")

plt.xlabel("Pressure (millibars)")

plt.ylabel("Humidity")

plt.plot(X, y_pred)
theta0 = pr_model.intercept_

_, theta1, theta2 = pr_model.coef_

theta0, theta1, theta2
# Predict humidity for a pressure of 1007 millibars

# Tranform 1007 to 1, 1007, 1007^2 suitable for input, using 

# pre_process.fit_transform

y_new = pr_model.predict(pre_process.fit_transform([[1007]]))

y_new
plt.scatter(X, y, c = "Black")

plt.xlabel("Pressure (millibars)")

plt.ylabel("Humidity")

plt.plot(X, y_pred)

plt.scatter(1007, y_new, c = "red")
from sklearn.metrics import mean_squared_error

mean_squared_error(y, y_pred)