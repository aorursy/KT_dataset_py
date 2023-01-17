import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from pandas import read_csv
filename = ("../input/housing.csv")

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

boston = read_csv(filename, delim_whitespace=True, names=names)
print(boston.shape)
boston.head()
# from sklearn.datasets import load_boston

# boston_dataset = load_boston()

# print(boston_dataset.keys())

# boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

# boston['MEDV'] = boston_dataset.target
boston.isnull().sum()
boston.corr()
plt.figure(figsize=(12,10))

sns.heatmap(boston.corr().round(2),cmap='coolwarm',annot=True)
corr = boston.corr()

corr['MEDV'].sort_values(ascending = False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].scatter(x=boston['LSTAT'],y=boston['MEDV'],color='red')

axes[0].set_ylabel('MEDV')

axes[0].set_xlabel('LSTAT')





axes[1].scatter(x=boston['RM'],y=boston['MEDV'])

axes[1].set_ylabel('MEDV')

axes[1].set_xlabel('RM')

X = boston[['LSTAT', 'RM']]

y = boston['MEDV']
X.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
lm = LinearRegression(normalize=True)
lm.fit(X_train,y_train)
print(lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics

from sklearn.metrics import r2_score

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
r2 = r2_score(y_test, predictions)
print('R2 score is {}'.format(r2))
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(2)

X_train_poly = poly_features.fit_transform(X_train)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

  
y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))

rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_predict))

r2_test = r2_score(y_test, y_test_predict)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
print('R2 score is {}'.format(r2_test))