# import libraries and functions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
# set matplotlib to inline
%matplotlib inline
# import our data
pak_data = pd.read_csv("../input/pakistan_data_1960_2017.csv")
# a look on how our data looks like
pak_data.head()
# a look on dataset dimensions
pak_data.shape
# now let's setup the variables ( features, labels) for our dataset

# we are taking year as a feature and other colomns as labels for now, but you can switch
X = pak_data[['year']]

y_gdp_usd = pak_data['gdp-usd']
y_military_expenditure_percent_gdp = pak_data['military-expenditure-percent-gdp']
y_urban_population = pak_data['urban-population']
y_rural_population = pak_data['rural-population']
y_life_expectancy_at_birth_male_years = pak_data['life-expectancy-at-birth-male-years']
y_life_expectancy_at_birth_female_years = pak_data['life-expectancy-at-birth-female-years']
y_life_expectancy_at_birth_total_years = pak_data['life-expectancy-at-birth-total-years']
y_co2_emissions_kt = pak_data['co2-emissions-kt']
# for this kernal lets use gdp per year
y = y_gdp_usd
# creating a train/test data is recommended 
# but we are skipping it for now as our dataset is small
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# now create a polynomial feature normalizer / model
poly_reg = PolynomialFeatures(degree=3)

# process our features for the normalization / polynomial regression
X_poly = poly_reg.fit_transform(X)
# see/test the output from polynomial function
poly_reg.fit(X_poly, y)
# now let's create a linear regression model
lm = LinearRegression()
# here we train our linear regression model by using the polynomial data
lm.fit(X_poly, y)

# predict the data from linear regression model by using polynomial model
predict = lm.predict(poly_reg.fit_transform(X))
# set some legend and labels for the visual plot
title = "Pakistan's Data Prediction"
x_label = "Year"
y_label = "GDP in USD$ Per Year"
# show the scatter plot along with the output/prediction from our model
fig = plt.figure(figsize=(15, 10))
plt.scatter(X, y, color='blue', label=y_label)
plt.plot(X, predict, color='red', label='POLYR')
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.legend()
plt.show()