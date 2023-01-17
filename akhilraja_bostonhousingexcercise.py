import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import datasets

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso





seed = 0

np.random.seed(seed)
from sklearn.datasets import load_boston
# Load the Boston Housing dataset from sklearn

boston = load_boston()

bos = pd.DataFrame(boston.data)

# give our dataframe the appropriate feature names

bos.columns = boston.feature_names

# Add the target variable to the dataframe

bos['Price'] = boston.target
# For student reference, the descriptions of the features in the Boston housing data set

# are listed below

boston.DESCR
bos.head()
# Select target (y) and features (X)

X = bos.iloc[:,:-1]

y = bos.iloc[:,-1]
# Split the data into a train test split

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
# Check the correlation 



corr_mat = x_train.corr()

plt.figure(figsize=(10,6))

sns.heatmap(corr_mat, annot=True)

plt.show()
# Linear Regression Model

lm = LinearRegression()

lm.fit(x_train, y_train)

y_pred = lm.predict(x_test)



r_squared_lm = r2_score(y_test, y_pred)

rmse_lm = np.sqrt(mean_squared_error(y_test, y_pred))

mae_lm = mean_absolute_error(y_test, y_pred)



#Print Metrics

print("Linear Regression")

print("R-squared = ", r_squared_lm)

print("RMSE = ", rmse_lm)

print("MAE = ", mae_lm)

# Ridge Regression



ridge = Ridge()

ridge.fit(x_train, y_train)

y_pred = lm.predict(x_test)



r_squared_ridge = r2_score(y_test, y_pred)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred))

mae_ridge = mean_absolute_error(y_test, y_pred)



#Print Metrics

print("Ridge Regression")

print("R-squared = ", r_squared_ridge)

print("RMSE = ", rmse_ridge)

print("MAE = ", mae_ridge)

# Lasso Regression



lasso = Lasso()

lasso.fit(x_train, y_train)

y_pred = lasso.predict(x_test)



r_squared_lasso = r2_score(y_test, y_pred)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred))

mae_lasso = mean_absolute_error(y_test, y_pred)



#Print Metrics

print("Lasso Regression")

print("R-squared = ", r_squared_lasso)

print("RMSE = ", rmse_lasso)

print("MAE = ", mae_lasso)


