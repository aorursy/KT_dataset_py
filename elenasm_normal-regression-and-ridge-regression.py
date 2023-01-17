

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# loading the Boston dataset

from sklearn.datasets import load_boston

house_price = load_boston()

df_labels = pd.DataFrame(house_price.target)

df = pd.DataFrame(house_price.data)

print(df_labels.head())

print(df.head())
df_labels.columns = ['PRICE']

df.columns = house_price.feature_names

print(df.shape)

print(df_labels.shape)
df_total = df.merge(df_labels, left_index = True, right_index = True)

df_total.head()


df_total.describe()

df_total.info()
plt.hist(df_labels['PRICE'], bins = 8)
from scipy.stats import skew,kurtosis 

print(skew(df_labels['PRICE']))

print(kurtosis(df_labels['PRICE'])) 
corr_matrix = df_total.corr(method = 'pearson')

corr_matrix 
# standardize and train/test split: standardize only data, not target

df = preprocessing.scale(df)

X_train, X_test, y_train, y_test = train_test_split(

    df, df_labels, test_size=0.3, random_state=10)
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)
#on train set

from sklearn.metrics import mean_squared_error

y_train_predicted = lin_reg.predict(X_train)

lin_mse = mean_squared_error(y_train_predicted, y_train)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
print(lin_reg.intercept_)

print(lin_reg.coef_)
#on test set

y_test_predicted = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test_predicted, y_test)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
#let's see how rmse compares to the rest of the target var desciptives

df_labels['PRICE'].describe()
#do the same with ridge

ridge_reg = Ridge(alpha=0)

ridge_reg.fit(X_train, y_train)

y_train_predicted = ridge_reg.predict(X_train)

ridge_mse = mean_squared_error(y_train_predicted, y_train)

ridge_rmse = np.sqrt(ridge_mse)

ridge_rmse 
ridge_reg = Ridge(alpha=50)

ridge_reg.fit(X_train, y_train)

y_train_predicted = ridge_reg.predict(X_train)

ridge_mse = mean_squared_error(y_train_predicted, y_train)

ridge_rmse = np.sqrt(ridge_mse)

ridge_rmse 
from sklearn.linear_model import RidgeCV

regr_cv = RidgeCV(alphas=[0.1,0.3, 0.5,0.7, 1.0, 10.0, 50.0])

model = regr_cv.fit(X_train, y_train)
model.alpha_
y_train_predicted = regr_cv.predict(X_train)

ridge_mse = mean_squared_error(y_train_predicted, y_train)

ridge_rmse = np.sqrt(ridge_mse)

ridge_rmse 
def function(i):

    ridge_reg = Ridge(alpha = i)

    ridge_reg.fit(X_train, y_train)

    y_train_predicted = ridge_reg.predict(X_train)

    ridge_mse = mean_squared_error(y_train_predicted, y_train)

    ridge_rmse = np.sqrt(ridge_mse)

    print(ridge_rmse)
function(0.1)
#on test set

y_test_predicted = ridge_reg.predict(X_test)

lin_mse = mean_squared_error(y_test_predicted, y_test)

lin_rmse = np.sqrt(lin_mse)

lin_rmse