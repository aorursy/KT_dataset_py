# https://www.w3schools.com/python/python_ml_multiple_regression.asp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Multiple Regression
# Multiple regression is like linear regression, but with more than one independent value,
# meaning that we try to predict a value based on two or more variables.
import pandas
from sklearn import linear_model
# Car, Model, Volume, Weight, CO2
df = pandas.read_csv("../input/linear-regression-dataset/cars.csv")
# print(df) # 0, Toyota, Aygo, 1000, 790, 99
X = df[['Weight', 'Volume']]
y = df['CO2']
regr = linear_model.LinearRegression()
# Tip: It is common to name the list of independent values with a upper case X,
# and the list of dependent values with a lower case y.
regr.fit(X, y)
# predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])
print("predictedCO2 [weight=2300kg, volume=1300ccm]:")
print(predictedCO2) # [107.2087328]

# Coefficient
# The coefficient is a factor that describes the relationship with an unknown variable.
print("Coefficient [weight, volume]")
print(regr.coef_) # [0.00755095 0.00780526]

df
# Result Explained
# The result array represents the coefficient values of weight and volume.
# Weight: 0.00755095
# Volume: 0.00780526
# These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
# And if the engine size (Volume) increases by 1 ccm, the CO2 emission increases by 0.00780526 g.
# I think that is a fair guess, but let test it!
# We have already predicted that if a car with a 1300ccm engine weighs 2300kg,
# the CO2 emission will be approximately 107g.
# What if we increase the weight with 1000kg?
predictedCO2 = regr.predict([[3300, 1300]])
print("predictedCO2 [weight=3300kg, volume=1300ccm]")
print(predictedCO2)