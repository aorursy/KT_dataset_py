import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import datasets
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
print(boston.DESCR)
bos.head()
# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]
# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
bos.dtypes
# Correlation threshold of 0.7
bos.corr()['Price'].abs() >= 0.7
# Correlation Plot
sns.set(rc={'figure.figsize':(12,9)})
correlation_matrix = bos.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
# Fit a linear regression model using OLS

from sklearn.linear_model import LinearRegression

slm = LinearRegression()
slm.fit(x_train[['RM','LSTAT']],
        y_train)
print(slm.intercept_)
print(slm.coef_)
y_pred_train = slm.predict(x_train[['RM','LSTAT']])
y_pred_test = slm.predict(x_test[['RM','LSTAT']])
from sklearn.metrics import r2_score, mean_squared_error
print('R2 score for train data: {}'.format(r2_score(y_train, y_pred_train)))
print('R2 score for test data: {}'.format(r2_score(y_test, y_pred_test)))
print('Mean squared error for train data: {}'.format(mean_squared_error(y_train, y_pred_train)))
print('Mean squared error for test data: {}'.format(mean_squared_error(y_test, y_pred_test)))
# Cross Validation Score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(slm, X[['RM','LSTAT']], y, cv=5, scoring='neg_mean_squared_error')
np.mean(np.sqrt(-scores))
bos.head()
plt.figure(figsize=(6,3))
plt.scatter(bos['RM'], bos['Price'])
plt.xlabel('RM')
plt.ylabel('Price')
rm_bucket = []
for i in list(bos['RM']):
    if i < 5:
        rm_bucket.append(1)
    elif i > 7:
        rm_bucket.append(2)
    else:
        rm_bucket.append(3)
        
bos['RM_bucket'] = rm_bucket
bos['RM_bucket'] = bos['RM_bucket'].astype('category')
X = bos.loc[:,bos.columns != 'Price']
y = bos.loc[:,'Price']

# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
x_train['RM_bucket'] = labelencoder.fit_transform(x_train['RM_bucket'])
x_test['RM_bucket'] = labelencoder.transform(x_test['RM_bucket'])
# Fit a linear regression model using OLS

from sklearn.linear_model import LinearRegression

slm = LinearRegression()
slm.fit(x_train[['RM','LSTAT','RM_bucket']],
        y_train)

print(slm.intercept_)
print(slm.coef_)

y_pred_train = slm.predict(x_train[['RM','LSTAT','RM_bucket']])
y_pred_test = slm.predict(x_test[['RM','LSTAT','RM_bucket']])
from sklearn.metrics import r2_score, mean_squared_error
print('R2 score for train data: {}'.format(r2_score(y_train, y_pred_train)))
print('R2 score for test data: {}'.format(r2_score(y_test, y_pred_test)))
print('Mean squared error for train data: {}'.format(mean_squared_error(y_train, y_pred_train)))
print('Mean squared error for test data: {}'.format(mean_squared_error(y_test, y_pred_test)))
plt.scatter(y_test, y_pred_test, marker='o')
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
