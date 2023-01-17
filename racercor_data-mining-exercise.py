# load libraries before using any function
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pylab 
import scipy.stats as stats
# read in the dataset and get columns name 
dataset = pd.read_csv('../input/kc_house_data.csv')
dataset.columns
# create new data frame by selecting some columns from original data set
new_dataset = dataset[['price','bedrooms','bathrooms','sqft_living','yr_built']]
# and obtain a summary of the new data frame 
new_dataset.info()
# get statistics and value counts to find outliers and potential bad data in the new data frame
new_dataset.describe()
# cleaning data by dropping rows for what the number of 'bedrooms' is equal to zero, considered as bad data
new_dataset = new_dataset[new_dataset.bedrooms != 0]
# cleaning data by dropping the row for what number of 'bedrooms' is below to '33', considered as potential oulier
new_dataset = new_dataset[new_dataset.bedrooms < 33]
# cleaning data by dropping rows for what the number of 'bathrooms' is equal to zero, considered as bad data
new_dataset = new_dataset[new_dataset.bathrooms != 0]
# cleaning data by dropping the row for what number of 'bathrooms' is below to '7', considered as potential oulier
new_dataset = new_dataset[new_dataset.bathrooms < 7]
# obtain a new summary of the data frame
new_dataset.info()
# get statistics and value counts to find outliers and potential bad data in the new data frame
new_dataset.describe()
# visually inspect the first five rows of the new data frame
new_dataset.head()
# draw a histogram plot based on' year built' of the new data frame
new_dataset.yr_built.hist()
# draw a scatter plot for each data frame based on 'number of bedrooms' to see the difference
dataset.plot(kind='scatter', x='bedrooms', y='price')
new_dataset.plot(kind='scatter', x='bedrooms', y='price')
# draw a scatter plot for each data frame based on 'number of bathrooms' to see the difference
dataset.plot(kind='scatter', x='bathrooms', y='price')
new_dataset.plot(kind='scatter', x='bathrooms', y='price')
# draw a scatter plot based on 'square footage of the home' of the new data frame
new_dataset.plot(kind='scatter', x='sqft_living', y='price')
# draw a matrix of scatter plots of the data set in a shape of 16" width x 16" height
# to roughly determine if there is a linear correlation between multiple variables
pd.plotting.scatter_matrix(new_dataset, figsize=(16,16))
# include the QQ norm to see if residuals are normal
residuals = np.random.normal(0,1, 1000)
# using 's' to get scaled line by the standard deviation and mean added) 
sm.qqplot(residuals, line='s')
pylab.show()
# construct the regression model using the Ordinary Least Squares (OLS) function
Y=new_dataset.price
X=new_dataset[['bedrooms','bathrooms','sqft_living','yr_built']]
X=sm.add_constant(X)
model=sm.OLS(Y,X).fit()
predictions=model.predict(X)
model.summary()
# Conclusions:
# house sale price is highly significanty affected by number of bedrooms, bathrooms, square footage of the home and year built.
# for each unit increase in the number of bedrooms the house sale price decreases by 75390.
# for each unit increase in the number of bathrooms the house sale price rises by 83680.
# for each squared feet more the house sale price increases by 300 dollars.