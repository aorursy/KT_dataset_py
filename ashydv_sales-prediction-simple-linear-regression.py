# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy and pandas package



import numpy as np

import pandas as pd



# Data Visualisation

import matplotlib.pyplot as plt 

import seaborn as sns
advertising = pd.DataFrame(pd.read_csv("../input/advertising.csv"))

advertising.head()
advertising.shape
advertising.info()
advertising.describe()
# Checking Null values

advertising.isnull().sum()*100/advertising.shape[0]

# There are no NULL values in the dataset, hence it is clean.
# Outlier Analysis

fig, axs = plt.subplots(3, figsize = (5,5))

plt1 = sns.boxplot(advertising['TV'], ax = axs[0])

plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])

plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])

plt.tight_layout()
# There are no considerable outliers present in the data.
sns.boxplot(advertising['Sales'])

plt.show()
# Let's see how Sales are related with other variables using scatter plot.

sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')

plt.show()
# Let's see the correlation between different variables.

sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)

plt.show()
X = advertising['TV']

y = advertising['Sales']
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
plt.scatter(X_train, y_train)

plt.plot(X_train, 6.948 + 0.054*X_train, 'r')

plt.show()
y_train_pred = lr.predict(X_train_sm)

res = (y_train - y_train_pred)
fig = plt.figure()

sns.distplot(res, bins = 15)

fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 

plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label

plt.show()
plt.scatter(X_train,res)

plt.show()
# Add a constant to X_test

X_test_sm = sm.add_constant(X_test)



# Predict the y values corresponding to X_test_sm

y_pred = lr.predict(X_test_sm)
y_pred.head()
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
#Returns the mean squared error; we'll take a square root

np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

r_squared
plt.scatter(X_test, y_test)

plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')

plt.show()