# Some usefull packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
# Importing dataset

df = pd.read_csv('../input/weatherww2/Summary of Weather.csv')



# Selecting min and max temperature columns

df = df[['MinTemp', 'MaxTemp']]

df.head()
# Scatter plot

plt.figure(figsize=(10, 5))

plt.scatter(df['MinTemp'], df['MaxTemp'],s=10)

plt.xlabel('Min Temperature °C',fontsize=15)

plt.ylabel('Max Temperature °C',fontsize=15)

plt.show()
# Drop anomalies data points

df.drop(df[(df['MinTemp'] < -15) & (df['MaxTemp'] > 15)].index, inplace = True)

df.drop(df[(df['MinTemp'] > 8) & (df['MaxTemp'] < -15)].index, inplace = True)



# Scatter plot after removing anomalies datapoint

plt.figure(figsize=(10, 5))

plt.scatter(df['MinTemp'], df['MaxTemp'],s=10)

plt.xlabel('Min Temperature °C',fontsize=15)

plt.ylabel('Max Temperature °C',fontsize=15)

plt.show()
# Scatter plot with few possible regression lines

plt.figure(figsize=(10, 5))

plt.scatter(df['MinTemp'], df['MaxTemp'],s=10)

plt.xlabel('Min Temperature °C',fontsize=15)

plt.ylabel('Max Temperature °C',fontsize=15)



x1 = [-38,35]

y1 = [-32,53]

plt.plot(x1, y1, color='orange')



x2 = [-38,35]

y2 = [-27,44]

plt.plot(x2, y2, color='red')

plt.show()
X = df['MinTemp']

y = df['MaxTemp']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
# Let's now take a look at the train dataset



X_train.head()
y_train.head()
import statsmodels.api as sm
# Add a constant to get an intercept

X_train_sm = sm.add_constant(X_train)



# Fit the resgression line using 'OLS'

lr = sm.OLS(y_train, X_train_sm).fit()
# Print the parameters, i.e. the intercept and the slope of the regression line fitted

lr.params
# Performing a summary operation lists out all the different parameters of the regression line fitted

print(lr.summary())
# Best fit line

plt.figure(figsize=(10, 5))

plt.scatter(X_train, y_train, s=10)

plt.plot(X_train, 10.6760 + 0.9201*X_train, 'r')

plt.xlabel('Min Temperature °C',fontsize=15)

plt.ylabel('Max Temperature °C',fontsize=15)

plt.show()
y_train_pred = lr.predict(X_train_sm)

res = (y_train - y_train_pred)
fig = plt.figure(figsize=(8, 4))

sns.distplot(res, bins = 15)

fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 

plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label

plt.show()
plt.figure(figsize=(8, 4))

plt.scatter(X_train,res)

plt.show()
# Add a constant to X_test

X_test_sm = sm.add_constant(X_test)



# Predict the y values corresponding to X_test_sm

y_pred = lr.predict(X_test_sm)
y_pred.head()
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
# Returns the root mean squared error

mean_squared_error(y_test, y_pred)
# Returns the mean squared error

np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

r_squared
plt.figure(figsize=(10, 5))

plt.scatter(X_test, y_test, s=10)

plt.plot(X_test, 10.6760 + 0.9201*X_test, 'r')

plt.show()
from sklearn.model_selection import train_test_split

X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)
X_train_lm.shape
X_train_lm = X_train_lm.values.reshape(-1,1)

X_test_lm = X_test_lm.values.reshape(-1,1)
print(X_train_lm.shape)

print(y_train_lm.shape)

print(X_test_lm.shape)

print(y_test_lm.shape)
from sklearn.linear_model import LinearRegression



# Representing LinearRegression as lr(Creating LinearRegression Object)

lm = LinearRegression()



# Fit the model using lr.fit()

lm.fit(X_train_lm, y_train_lm)
print(lm.intercept_)

print(lm.coef_)