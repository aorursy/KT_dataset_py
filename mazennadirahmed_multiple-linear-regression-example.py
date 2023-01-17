# Import Pandas Library, used for data manipulation

# Import matplotlib, used to plot our data

# Import numpy for linear algebra operations

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


# Import our WeatherDataM and store it in the variable weather_data_m

weather_data_m = pd.read_csv("../input/weatherdatam/WeatherDataM.csv") 

# Display the data in the notebook

weather_data_m
# Set the features of our model, these are our potential inputs

weather_features = ['Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)']



# Set the variable X to be all our input columns: Temperature, Wind Speed and Pressure

X = weather_data_m[weather_features]



# set y to be our output column: Humidity

y = weather_data_m.Humidity





# plt.subplot enables us to plot mutliple graphs

# we produce scatter plots for Humidity against each of our input variables



plt.subplot(2,2,1)

plt.scatter(X['Temperature (C)'],y)

plt.subplot(2,2,2)

plt.scatter(X['Wind Speed (km/h)'],y)

plt.subplot(2,2,3)

plt.scatter(X['Pressure (millibars)'],y)
# Pressure (millibars) is the column we wish to drop and 1 represent the axis number. We use 1 for columns and 0 for rows:



X = X.drop("Pressure (millibars)", 1)

X
# Import library to produce a 3D plot

from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



x1 = X["Temperature (C)"]

x2 = X["Wind Speed (km/h)"]



ax.scatter(x1, x2, y, c='r', marker='o')



# Set axis labels

ax.set_xlabel('Temperature (C)')

ax.set_ylabel('Wind Speed (km/h)')

ax.set_zlabel('Humidity')
from sklearn.linear_model import LinearRegression



mlr_model = LinearRegression()



# Fit our linear regression model to our data, we are essentially finding θ₀, θ₁ and θ₂ in our regression line: ŷ = θ₀– θ₁𝑥¹- θ₂𝑥²

mlr_model.fit(X, y)
y_pred = mlr_model.predict([[15, 21]])

y_pred