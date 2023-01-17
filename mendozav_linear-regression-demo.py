import numpy as np # linear algebra

import matplotlib.pyplot as plt



from sklearn import datasets, linear_model
# house price array

house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]

# house size array

size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700] 
# reshape the input to your regression

# formats array 

size2 = np.array(size).reshape((-1, 1))
regr = linear_model.LinearRegression()

# fits data to linear regression model

regr.fit(size2, house_price)

print("Coefficients: \n", regr.coef_)

print("Intercept: \n", regr.intercept_)
# testing out predictions

# use to predict the price of real estate by entering the size of a house

size_new = 1400

price = (size_new * regr.coef_) + regr.intercept_

print(price)

# gives the same answer just another way of printing

# print(regr.predict([[size_new]]))
# formula obtained for the trained model

def graph(formula, x_range):

    x = np.array(x_range)

    y = eval(formula)

    plt.plot(x, y)
# plotting the prediction line

graph('regr.coef_ * x + regr.intercept_', range(1000, 2700))

plt.scatter(size, house_price, color = 'black')

# labeling the axis

plt.ylabel('House Price')

plt.xlabel('Size of House')

plt.show()