#Import Libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# Load data        

house = pd.read_csv("../input/uci-ml-datasets/hou_all.csv")

house.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV','BIAS_COL']

print(house.shape)

house.head()
house = house.drop(['BIAS_COL'], axis=1)
# No missing value

house.isnull().sum()
house.info()
house.describe()
# set the size of the figure

sns.set(rc={'figure.figsize':(12,8)})



sns.distplot(house['MEDV'], bins=25, axlabel='House Price')
# Pairwise Correlational Matrix of house data with round upto 2 digits after decimal

correlational_matrix = house.corr().round(2)

#print(correlational_matrix)
#Heatmap

sns.heatmap(data=correlational_matrix, center=None, annot=True, linewidths=0, linecolor='white')
# Correlation -->

# MEDV is highly correlated with LSTAT and RM

# DIS and INDUS 

# DIS and NOX

# DIS and AGE

# TAX and RAD
f = plt.figure(figsize=(20,14))

gs = f.add_gridspec(2, 2)



sns.axes_style("darkgrid")



ax = f.add_subplot(gs[0,0])

x = house['LSTAT']

y = house['MEDV']

plt.scatter(x,y,marker='o')

plt.title('MEDV vs LSTAT')

plt.xlabel('LSTAT')

plt.ylabel('MEDV')



ax = f.add_subplot(gs[0,1])

x = house['RM']

y = house['MEDV']

plt.scatter(x,y,marker='o')

plt.title('MEDV vs RM')

plt.xlabel('RM')

plt.ylabel('MEDV')

f = plt.figure(figsize=(20,14))

gs = f.add_gridspec(2, 2)



sns.axes_style("darkgrid")



ax = f.add_subplot(gs[0,0])

x = house['INDUS']

y = house['DIS']

plt.scatter(x,y,marker='o')

plt.title('DIS vs INDUS')

plt.xlabel('INDUS')

plt.ylabel('DIS')



ax = f.add_subplot(gs[0,1])

x = house['NOX']

y = house['DIS']

plt.scatter(x,y,marker='o')

plt.title('DIS vs NOX')

plt.xlabel('NOX')

plt.ylabel('DIS')



ax = f.add_subplot(gs[1,0])

x = house['AGE']

y = house['DIS']

plt.scatter(x,y,marker='o')

plt.title('DIS vs AGE')

plt.xlabel('AGE')

plt.ylabel('DIS')



ax = f.add_subplot(gs[1,1])

x = house['TAX']

y = house['RAD']

plt.scatter(x,y,marker='o')

plt.title('RAD vs TAX')

plt.xlabel('TAX')

plt.ylabel('RAD')

# Drop features that are not much correlated with MEDV

house = house.drop(['DIS', 'CRIM', 'ZN', 'CHAS', 'AGE','RAD', 'B'], axis=1)

house.head()
y = house['MEDV']

X = house.drop(['MEDV'], axis=1)
X.head()
# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Fitting and traing data

lr = LinearRegression()

lr.fit(X_train, y_train)



y_predict = lr.predict(X_test)
# Metric Results

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_predict)))

print("R2:", r2_score(y_test, y_predict))
# Plot between predicted and actual values

plt.scatter(y_test, y_predict)

plt.show()
# Polynomial Regression



poly_features = PolynomialFeatures(degree=2)



X_train_poly = poly_features.fit_transform(X_train)

print(X_train_poly.shape)



pr = LinearRegression()

pr.fit(X_train_poly, y_train)



X_test_poly = poly_features.fit_transform(X_test)

y_predict_poly = pr.predict(X_test_poly)

# Metric Results better than Linear Regression

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_predict_poly)))

print("R2:", r2_score(y_test, y_predict_poly))
plt.scatter(y_test, y_predict_poly)

plt.show()