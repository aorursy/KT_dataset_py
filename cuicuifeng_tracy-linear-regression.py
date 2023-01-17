import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Housing = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')
Housing.info()
Housing.head()
Housing.describe()
Housing.columns
sns.pairplot(Housing)
sns.distplot(Housing['Price'])
sns.heatmap(Housing.corr())
Housing.columns
## First split up data into X array that contains the features to train on, and y array with the target variable.

## I will toss out the Address column because it onlu has text info that th elinear regression model can not use.

X = Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]

y = Housing['Price']
## Split the data into a training set and a testing set.

## I will train model on the training set and then use the test set to evaluate the model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
## Check out coefficients and how we can interpret them.

print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
## Residual Histogram

sns.distplot((y_test-predictions),bins=50)
from sklearn import metrics

print('MAE', metrics.mean_absolute_error(y_test, predictions))

print('MSE', metrics.mean_squared_error(y_test, predictions))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predictions)))