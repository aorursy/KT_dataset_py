# Import Pandas Library, used for data manipulation
# Import matplotlib, used to plot our data
# Import numpy for linear algebra operations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import our WeatherData and store it in the variable weather_data, ensure the correct filepath is used
weather_data = pd.read_csv("../input/weather-data/WeatherData.csv") 
# Display the data in the notebook
weather_data

# Set x to our input Temperature
x = weather_data['Temperature(C)']
# Set y to our output humidity, this is the value we trying to predict
y = weather_data['Humidity']
# Plot Humidity against Temperature
plt.scatter(x, y)
# Add labels to the graph
plt.xlabel('Temperature(C)')
plt.ylabel('Humidity')
from sklearn.linear_model import LinearRegression
# Setting x and y to the appropriate variables, we reshape x to turn it from a 1D array to a 2D array, ready to be used in our model.
x = weather_data[['Temperature(C)']]
y = weather_data['Humidity']
# Define the variable lr_model as our linear regression model
lr_model = LinearRegression()
# Fit our linear regression model to our data, we are essentially finding θ₀ and θ₁ in our regression line: ŷ = θ₀ + θ₁𝑥. Gradient descent and other methods are used for this.
lr_model.fit(x, y)
# Find predicted values for all x values by applying ŷᵢ = θ₀ + θ₁𝑥ᵢ
y_pred = lr_model.predict(x)
plt.scatter(x, y)
plt.xlabel('Temperature(C)')
plt.ylabel('Humidity')
# Here we are plotting our regression line ŷ = θ₀ + θ₁𝑥
           
plt.plot(x, y_pred)
# input 32 into our regression model "np.array([[32]]) reshapes 32 into a 2D array. This is done as our model only accepts inputs of this form.
y_pred = lr_model.predict([[32]])
y_pred