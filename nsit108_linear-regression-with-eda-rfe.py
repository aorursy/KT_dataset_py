# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read the given CSV file, and view some sample records

advertising = pd.read_csv("/kaggle/input/sales-advertisment/advertising.csv")
advertising.head()
#inspect the various aspects of our dataframe
advertising.shape
advertising.info()
advertising.describe()
import matplotlib.pyplot as plt 
import seaborn as sns
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', aspect=1, kind='scatter')
plt.show()
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
plt.plot(X_train, 6.9487 + 0.0545*X_train, 'r')
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
sns.pairplot(advertising)
plt.show()
plt.figure(figsize=(20, 15))
plt.subplot(2,2,1)
sns.boxplot(y= advertising['TV'],   palette="Set1"   )
plt.subplot(2,2,2)
sns.boxplot(y = advertising['Radio'],   palette="Set2" )
plt.subplot(2,2,3)
sns.boxplot(y= advertising['Newspaper'],   palette="Set3" )
plt.subplot(2,2,4)
sns.boxplot(y= advertising['Sales'],  palette="Set1")
plt.show()

sns.boxplot(data=advertising.iloc[:,0:4])
plt.show()
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(advertising, train_size = 0.7, test_size = 0.3, random_state = 100)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['TV','Radio','Newspaper','Sales']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (6, 3))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()
y_train = df_train.pop('Sales')
X_train = df_train
X_train.head()
import statsmodels.api as sm

# Add a constant
X_train_lm = sm.add_constant(X_train )

# Create a first fitted model
lr = sm.OLS(y_train, X_train_lm).fit()
# Check the parameters obtained

lr.params
# Print a summary of the linear regression model obtained
print(lr.summary())
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables and insignificant variables

X_train2 = X_train.drop('Newspaper', 1,)
X_train2.head()
# Build a second fitted model
X_train_lm = sm.add_constant(X_train2)

lr_2 = sm.OLS(y_train, X_train_lm).fit()
# Print the summary of the model
print(lr_2.summary())
# Calculate the VIFs again for the new model

vif = pd.DataFrame()
vif['Features'] = X_train2.columns
vif['VIF'] = [variance_inflation_factor(X_train2.values, i) for i in range(X_train2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_price = lr_2.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test.head()
num_vars = ['TV', 'Radio', 'Sales'  ]

#df_test[num_vars] = scaler.transform(df_test[num_vars])


df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
df_test.describe()
#### Dividing into X_test and y_test
y_test = df_test.pop('Sales')
X_test = df_test
# Adding constant variable to test dataframe
X_test_m4 = sm.add_constant(X_test)
X_test_m4.head()
# Creating X_test_m4 dataframe by dropping variables from X_test_m4

X_test_m4 = X_test_m4.drop(["Newspaper" ], axis = 1)
# Making predictions using the fourth model

y_pred_m4 = lr_2.predict(X_test_m4)
X_test.head()
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_m4)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)      
 
plt.show()
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(advertising, train_size = 0.7, test_size = 0.3, random_state = 100)
y_train = df_train.pop('Sales')
X_train = df_train
X_train  = X_train.drop('Newspaper', 1,)
X_train.head()
# Add a constant
X_train_lm = sm.add_constant(X_train )
# Create a first fitted model
lr_3 = sm.OLS(y_train, X_train_lm).fit()
# Check the parameters obtained

lr_3.params

# Print a summary of the linear regression model obtained
print(lr_3.summary())
y_train_price = lr_3.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
# Dividing into X_test and y_test
y_test = df_test.pop('Sales')
X_test = df_test
# Adding constant variable to test dataframe
X_test_m3 = sm.add_constant(X_test)
# Creating X_test_m4 dataframe by dropping variables from X_test_m4

X_test_m3 = X_test_m3.drop(["Newspaper" ], axis = 1)
# Making predictions using the fourth model

y_pred_m3 = lr_3.predict(X_test_m3)
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_m4)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)      
