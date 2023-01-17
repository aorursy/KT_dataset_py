import pandas as pd

# read CSV file from the 'data' subdirectory using a relative path
data = pd.read_csv('../input/advertising.csv', index_col=0)

# display the first 5 rows
data.head()
# display the last 5 rows
data.tail()
# check the shape of the DataFrame (rows, columns)
data.shape
#conventional way to import seaborn
import seaborn as sns

# allow plots to appear within the notebook
%matplotlib inline
# visualize the realtioship betwwen the features and the response using scatterplots
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
# create a Python list of feature names
feature_cols = ['TV', 'Radio', 'Newspaper']

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# equivalent command to do this in one line
X = data[['TV','Radio', 'Newspaper']]

# Print the first 5 rows
X.head()
# check the type and the shape of X
print(type(X))
print(X.shape)
# select a Series from the DataFrame
y = data['Sales']

# equivalent command that works if there are no spaces in the column name
y = data.Sales

# print the first 5 values
y.head()
# check the type and the shape of y
print(type(y))
print(y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# default split is 75% for training and 25% for testing
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# import model
from sklearn.linear_model import LinearRegression

# instantiate 
linreg = LinearRegression()

# fit the model to the training data
linreg.fit(X_train, y_train)
# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)
# pair the feature names with the coefficients
list(zip(feature_cols, linreg.coef_))
# make predictions on the testing set
y_pred = linreg.predict(X_test)
y_pred
# define true and predicted response values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
# calculate MAE by hand
print((10 + 0 + 20 + 10)/4.)

# calculate MAE using scikit-learn
from sklearn import metrics
print(metrics.mean_absolute_error(true,pred))
# calculate MSE by hand
print((10**2 + 0**2 + 20**2 + 10**2)/4.)

# calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))
# calculate RMSE by hand 
import numpy as np
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))

# calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))
print(np.sqrt(metrics.mean_squared_error(y_test,  y_pred)))
# create a Python list of feature names
feature_cols = ['TV', "Radio"]

# use the list to select a subset of the original DataFrame
X = data[feature_cols]

# select a Series from the Datarame
y = data.Sales

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#fit the model to the training data (learn from coefficients)
linreg.fit(X_train, y_train)

# make predicitons on the testing set
y_pred = linreg.predict(X_test)

# comopute thr RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))