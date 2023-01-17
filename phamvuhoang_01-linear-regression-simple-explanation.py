import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data_path = '/kaggle/input/housingprice/ex1data2.txt'



data = pd.read_csv(data_path, sep = ',', header = None)

data.columns = ['Living Area', 'Bedrooms', 'Price']
# Print out first 5 rows to get the imagination of data

data.head()
# Is there any missing data or not?

data.isnull().values.any()
data[['Bedrooms']].plot(kind = 'hist', bins = [0, 1, 2, 3, 4, 5, 6], rwidth = 0.8)
data[['Living Area']].plot(kind = 'hist', rwidth = 0.8)
data[['Price']].plot(kind = 'hist', rwidth = 0.8)
data.hist(rwidth = 1)
data.groupby('Bedrooms')['Price'].nunique().plot(kind = 'bar')
data.plot(kind='scatter', x = 'Bedrooms', y = 'Price', color = 'green')
data.plot(kind = 'scatter', x = 'Living Area', y = 'Bedrooms', color = 'blue')
data.plot(kind='scatter', x = 'Living Area', y = 'Price', color = 'green')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
def plot_3d(plt, x, y, z, color = 'blue'):

    _3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

    _3d_figure.plot(X0, Y0, Z0, color = color)



    plt.show()   
X0 = data['Bedrooms']

Y0 = data['Living Area']

Z0 = data['Price']



plot_3d(plt, X0, Y0, Z0)
X1 = range(data.shape[1])

Y1 = range(data.shape[0])

X1, Y1 = np.meshgrid(X1, Y1)



_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot_wireframe(X1, Y1, data)



plt.show()
from sklearn.linear_model import LinearRegression
X = data['Bedrooms'].values.reshape(-1, 1)

Y = data['Price'].values.reshape(-1, 1)



# Visualize the data

plt.scatter(X, Y)



# Train the model

model = LinearRegression()

model.fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression predict result

plt.plot(X, Y_pred, color = 'red')



plt.show()
print('theta_0: ', model.intercept_)

print('theta_1 ', model.coef_)
print('Coeficient of Detemination: ', model.score(X, Y))
X = np.array(data[['Living Area']])

Y = np.array(data['Price'])



# Visualize the data

plt.scatter(X, Y)



# Train the model

model = LinearRegression()

model.fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression predict result

plt.plot(X, Y_pred, color = 'green')



plt.show()
print('theta_0: ', model.intercept_)

print('theta_1 ', model.coef_)
print('Coeficient of Detemination: ', model.score(X, Y))
X_test = [[3000]]

Y_test = model.predict(X_test)



print(Y_test)
X = np.array(data[['Bedrooms', 'Living Area']])

Y = np.array(data['Price'])



# Train the model

model = LinearRegression()

model.fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



# Draw the linear regression predict result

_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot(X[:, 0], X[:, 1], Y, color = 'green')

_3d_figure.plot(X[:, 0], X[:, 1], Y_pred, color = 'red')
print('theta_0: ', model.intercept_)

print('theta_1 ', model.coef_[0])

print('theta_2 ', model.coef_[1])
print('Coeficient of Detemination: ', model.score(X, Y))
X_test = [[1, 3000]]

Y_test = model.predict(X_test)



print(Y_test)
X_test = [[2, 3000]]

Y_test = model.predict(X_test)



print(Y_test)
X_test = [[3, 3000]]

Y_test = model.predict(X_test)



print(Y_test)
X_test = [[100, 3000]]

Y_test = model.predict(X_test)



print(Y_test)
from sklearn.preprocessing import PolynomialFeatures
X = np.array(data[['Living Area']])

Y = np.array(data['Price'])



# degree is an integer (2 by default) that represents the degree of the polynomial regression function

# interaction_only is a Boolean (False by default) that decides whether to include only interaction features (True) or all features (False).

# include_bias is a Boolean (True by default) that decides whether to include the bias (intercept) column of ones (True) or not (False).

transfomer = PolynomialFeatures(degree = 2, include_bias = True)

transfomer.fit(X)



# Create new-transformed input

X_ = transfomer.transform(X)
print('Before and after transformation: ', X.shape, X_.shape)
# Train the model

model = LinearRegression()

model.fit(X_, Y)



# Predict with the same input data

Y_pred = model.predict(X_)
# Visualize the data

plt.scatter(X, Y)



# Draw the linear regression predict result

plt.plot(X, Y_pred, color = 'red')



plt.show()
print('Coeficient of Detemination: ', model.score(X_, Y))
theta_0 = model.intercept_

theta_1 = model.coef_[0]

theta_2 =  model.coef_[1]



print('theta_0: ', theta_0)

print('theta_1: ', theta_1)

print('theta_2: ', theta_2)
import math

def graph(formula, x_range):  

    x = np.array(x_range)  

    y = eval(formula)

    plt.plot(x, y)  

    plt.show()



fomular = str(theta_2) + '*x**2+' + str(theta_1) + '*x+' + str(theta_0)

graph(fomular, range(math.ceil(X.max())))
X = np.array(data[['Living Area', 'Bedrooms']])

Y = np.array(data['Price'])



transfomer = PolynomialFeatures(degree = 2, include_bias = True)

transfomer.fit(X)



# Create new-transformed input

X_ = transfomer.transform(X)
print('Before and after transformation: ', X.shape, X_.shape)
# Train the model

model = LinearRegression()

model.fit(X_, Y)



# Predict with the same input data

Y_pred = model.predict(X_)
# Visualize the data

_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot(X[:, 1], X[:, 0], Y, color = 'green')

_3d_figure.plot(X[:, 1], X[:, 0], Y_pred, color = 'red')
print('Coeficient of Detemination: ', model.score(X_, Y))
print('theta_0 | 1 | 2 | 3 | 4 | 5 | 6')

print(model.intercept_, ' | ', model.coef_[0], ' | ', model.coef_[1], ' | ', model.coef_[2], ' | ', model.coef_[3], ' | ', model.coef_[4], ' | ', model.coef_[5])
X_test = np.array([[3000, 1]])

X_test_ = transfomer.transform(X_test)

X_test_.shape

Y_test = model.predict(X_test_)



print(Y_test)
def PolyCoefficients(x, coeffs):

    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.



    The coefficients must be in ascending order (``x**0`` to ``x**o``).

    """

    o = len(coeffs)

    print(f'# This is a polynomial of order {ord}.')

    y = 0

    for i in range(o):

        y += coeffs[i]*x**i

    return y



coeffs = np.append(np.array(model.intercept_), model.coef_)

plt.plot(X, PolyCoefficients(X, coeffs))

plt.show()