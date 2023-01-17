import numpy as np # Mathematical Operation tool
import pandas as pd # Data Manipulation tool
import matplotlib.pyplot as plt # Data Visualization tool
# displaying plot in notebook
%matplotlib inline 
# Loading the data
data = pd.read_csv('../input/linear-regression-dataset/Linear Regression - Sheet1.csv')
#printing the head
data.head()
# Plotting the graph
plt.scatter(data['X'],data['Y'])
# Looking for outliers
data.tail()
#Removing the outliers
new_data = data[:data.shape[0]-2].copy()
new_data
plt.scatter(new_data['X'],new_data['Y'])
# Defing the dependent and independent varibale
x1 = new_data['X'] #Indepnedent variable
y = new_data['Y'] #dependent variable
import statsmodels.api as sm

x = sm.add_constant(x1)  # adding const to dependent variable
result = sm.OLS(y,x).fit()  #Fitting our model
result.summary()
# Definng the Linear Regression
c = 3.2222 # Intercept
m = 0.6667  # coefficient

yhat = m*x1 + c
# Plotting the regression line
plt.scatter(new_data['X'],new_data['Y'])
plt.plot(new_data['X'],yhat,label='LinearRegression',c='Red') # Regression Line
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')