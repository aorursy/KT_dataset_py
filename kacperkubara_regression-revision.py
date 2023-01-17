import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualisations



import os

print(os.listdir("../input")) # print all the files from the "input" directory (we provided this directory! When using locally inject path to your files) 
dataset_2015 = pd.read_csv("../input/2015.csv") # Read csv file

dataset_2016 = pd.read_csv("../input/2016.csv")

dataset_2017 = pd.read_csv("../input/2017.csv")



dataset_2015 = dataset_2015.sample(frac=1).reset_index(drop=True) # randomly shuffle the rows (for the purpose of our exercise later)

dataset_2016 = dataset_2016.sample(frac=1).reset_index(drop=True)

dataset_2017 = dataset_2017.sample(frac=1).reset_index(drop=True)
pd.set_option('display.max_columns', None) #  Ensures that all columns will be displayed

print(dataset_2015.head(3)) # Prints first 3 entries

print('\n\n')



print('COLUMN NAMES')

print(dataset_2015.columns) # Prints the column names

print("NUMBER OF COLUMNS: " + str(len(dataset_2015.columns))) # Prints no. columns
import seaborn as sns # Library for more fancy plots

corr= dataset_2015.corr() # Creates correlation matrix

sns.heatmap(corr, xticklabels=corr.columns.values,

            yticklabels=corr.columns.values) # Creats a heatmap based on correlation matrix
# Write the code here
# Choosing variable to be predicted

y = dataset_2015['Happiness Score'] # Happiness Score chosen as a value to be predicted
# Write the code here
# Split data into train and test dataset

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# Scale/Normalise the data

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

sc_y = StandardScaler()



# Unscaled data - we will need it for Random Forest Regression and plotting the LinearRegression model

x_train_unscaled = x_train.copy()

x_test_unscaled  = x_test.copy()

y_train_unscaled = y_train.copy()



# Scale the data

x_train = sc_x.fit_transform(x_train)

x_test  = sc_x.transform(x_test)

y_train = y_train.values.reshape(-1, 1)

y_train = sc_y.fit_transform(y_train)



print('SCALED X_TRAIN:')

print(x_train[:5])

print('\nSCALED Y_TRAIN:')

print(y_train[:5])
# Train Model

from sklearn.linear_model import LinearRegression

regr = LinearRegression()

regr.fit(x_train, y_train) # training the model - simple as that huh



# Predict Results

y_pred = regr.predict(x_test)

y_pred = sc_y.inverse_transform(y_pred)

print('PREDICTED VALUES (UNSCALED): ')

print(y_pred[:5])
# Measure Accuracy

from sklearn.metrics import mean_squared_error

acc = mean_squared_error(y_test, y_pred) # Mean Squared Error to measure the accuracy

print('ACCURACY(MSE): ')

print(acc)
plt.title("Linear Regression Prediction")

plt.xlabel("Some Feature")

plt.ylabel("Happiness Score")

# Values in blue are those predicted by the model

plt.scatter(x_test_unscaled['Family'], y_pred, color = 'b')

# Values in red are orignal dataset points

plt.scatter(x_test_unscaled['Family'], y_test, color = 'r') 

# Display graph

plt.show()
# Import libraries for the Polynomial Regression model

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



# Creates separate variables for each of the polynomial degree of the feature 

poly_reg = PolynomialFeatures(degree = 3)

x_poly_train = poly_reg.fit_transform(x_train)

x_poly_test  = poly_reg.fit_transform(x_test)

poly_reg.fit(x_poly_train, y_train)



# Fit the polynomial features to the LinearRegression model

lin_reg = LinearRegression()

lin_reg.fit(x_poly_test, y_test)



# Predict the result

y_pred = lin_reg.predict(x_poly_test)



# Unscale the data

y_pred = sc_y.inverse_transform(y_pred)

print('PREDICTED VALUES (UNSCALED): ')

print(y_pred[:5])
# Measure Accuracy

from sklearn.metrics import mean_squared_error

acc = mean_squared_error(y_test, y_pred)



print('ACCURACY(MSE): ')

print(acc)
# Train Model

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators = 10, random_state = 42)



# For the RandomForest we don't have to scale the data

# which is due to the different algorithm being used

# Use x_train_unscaled, y_train_unscaled and x_test_unscaled instead!



# Fit the model



# Predict Results



# Predict the values

# Measure Accuracy

from sklearn.metrics import mean_squared_error

acc = mean_squared_error(y_test, y_pred)



print('ACCURACY(MSE): ')

print(acc)