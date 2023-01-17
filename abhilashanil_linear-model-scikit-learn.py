# Import required packages

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn

import sklearn.linear_model
# Load the data and check the shape

oecd_bli = pd.read_csv('../input/oecd_bli_2015.csv', thousands=',')

gdp_per_capita = pd.read_csv('../input/gdp_per_capita.csv', encoding='latin1', na_values='n/a')



print (oecd_bli.shape)

print (gdp_per_capita.shape)

print (oecd_bli.head(3))

print (gdp_per_capita.head(3))



# For simplicity and brevity's sake considering Life expectancy

# as the only factor that determines happiness

oecd_bli = oecd_bli[(oecd_bli['Inequality'] == 'Total') &

                    (oecd_bli['Indicator'] == 'Life expectancy')]

# Prepare the data

combined_data = pd.merge(gdp_per_capita, oecd_bli, on=['Country'])



gdp_value = combined_data[['2015']].copy()

bli_value = combined_data[['Value']].copy()



gdp_value.columns = ['GDP per capita']

bli_value.columns = ['Life satisfaction']



country_stats = pd.concat([gdp_value, bli_value], axis=1)



print (country_stats)



X = np.c_[country_stats['GDP per capita']]

y = np.c_[country_stats['Life satisfaction']]
# Visualize the data

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')

plt.show()
# Select a linear model

model = sklearn.linear_model.LinearRegression()



# Train the model

model.fit(X, y)



# Make a prediction for Cyprus

X_new = [[101994]]



print (model.predict(X_new))