# For linear algebra,
import numpy as np

# Data
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Model building and helper libraries
import xgboost
import math
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import model_selection, tree, linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
# Read the data into a data frame
data = pd.read_csv('../input/kc_house_data.csv')
# Check the number of data points in the data set
print(len(data))
# Check the number of features in the data set
print(len(data.columns))
# Check the data types
print(data.dtypes.unique())
# Since there are Python objects in the data set, we may have some categorical features. Let's check them.
data.select_dtypes(include=['O']).columns.tolist()
# Check number of columns with NaN
print(data.isnull().any().sum(), ' / ', len(data.columns))
# Check number of data points with NaN
print(data.isnull().any(axis=1).sum(), ' / ', len(data))
# Independent variables also known as features
features = data.iloc[:,3:].columns.tolist()
# Dependent Variables also known as target
target = data.iloc[:,2].name
# Dictionary to store correlations key: feature_name, value: correlation between feature and target
correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]
y = data.loc[:,['sqft_living','grade',target]].sort_values(target, ascending=True).values
x = np.arange(y.shape[0])
%matplotlib inline
plt.subplot(3,1,1)
plt.plot(x,y[:,0])
plt.title('Sqft and Grade vs Price')
plt.ylabel('Sqft')

plt.subplot(3,1,2)
plt.plot(x,y[:,1])
plt.ylabel('Grade')

plt.subplot(3,1,3)
plt.plot(x,y[:,2],'r')
plt.ylabel("Price")

plt.show()
# Train a linear regression model
regr = linear_model.LinearRegression()
new_data = data[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view',
                 'sqft_basement','lat','waterfront','yr_built','bedrooms']]
# X -> Independent variables
# y -> Dependent variable
X = new_data.values
y = data.price.values
# Splitting the data into train and test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.2, 
                                                                    random_state=4)
# Training the model
regr.fit(X_train, y_train)
# Predicting on test data
predictions = regr.predict(X_test)
print(predictions)
print(f'Mean Squared Error: {metrics.mean_squared_error(predictions,y_test)}')
print(f'Mean Absolute Error: {metrics.mean_absolute_error(predictions,y_test)}')
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(predictions,y_test))}')
# Let's try XGboost algorithm to see if we can get better results
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
traindf, testdf = train_test_split(X_train, test_size = 0.3)
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)
print(f'Mean Squared Error: {metrics.mean_squared_error(predictions,y_test)}')
print(f'Mean Absolute Error: {metrics.mean_absolute_error(predictions,y_test)}')
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(predictions,y_test))}')