import sys

print(sys.version)
#Load sample data



x = [4, 8, 12, 25, 32, 43, 58, 63, 69, 79]

y = [20, 33, 50, 56, 42, 31, 33, 46, 65, 75]


%matplotlib inline

from matplotlib import pyplot as plt



plt.scatter(x, y)
#Define a quadratic polynomial function and a loss function





def func(p, x):

    """Define a quadratic polynomial function

    """

    w0, w1, w2 = p

    f = w0 + w1*x + w2*x*x

    return f



def err_func(p, x, y):

    """Define a loss function

    """

    ret = func(p, x) - y

    return ret
#Type Your Code Here -

import numpy as np



p_init = np.random.randn(3) # Generate 3 random numbers



p_init
#Type Your Code Here -

"""

Use the least squares function provided by SciPy

"""



from scipy.optimize import leastsq



parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))



print('Fitting Parameters: ', parameters[0])
"""

Draw the fitted figure

"""



#Generate the testing points

x_temp = np.linspace(0, 80, 10000)



# Draw the fitted curve

plt.plot(x_temp, func(parameters[0], x_temp), 'r')



# Draw the original scatters

plt.scatter(x, y)
"""

N-degree Polynomial Fitting

"""



def fit_func(p, x):

    """Define an N-degree polynomial function

    """

    f = np.poly1d(p)

    return f(x)



def err_func(p, x, y):

    """Define a loss function

    """

    ret = fit_func(p, x) - y

    return ret



def n_poly(n):

    """N-degree polynomial fitting

    """

    p_init = np.random.randn(n) # Generate N random numbers

    parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))

    return parameters[0]
#Type Your Code Here -

n_poly(3)
"""

Plot the fitting result of the 4-, 5-, 6-, 7- and 8- degree polynomials

"""



#Generate the testing points

x_temp = np.linspace(0, 80, 10000)



#Generate the sub-images

fig, axes = plt.subplots(2, 3, figsize=(15,10))



axes[0,0].plot(x_temp, fit_func(n_poly(4), x_temp), 'r')

axes[0,0].scatter(x, y)

axes[0,0].set_title("m = 4")



axes[0,1].plot(x_temp, fit_func(n_poly(5), x_temp), 'r')

axes[0,1].scatter(x, y)

axes[0,1].set_title("m = 5")



axes[0,2].plot(x_temp, fit_func(n_poly(6), x_temp), 'r')

axes[0,2].scatter(x, y)

axes[0,2].set_title("m = 6")



axes[1,0].plot(x_temp, fit_func(n_poly(7), x_temp), 'r')

axes[1,0].scatter(x, y)

axes[1,0].set_title("m = 7")



axes[1,1].plot(x_temp, fit_func(n_poly(8), x_temp), 'r')

axes[1,1].scatter(x, y)

axes[1,1].set_title("m = 8")



axes[1,2].plot(x_temp, fit_func(n_poly(9), x_temp), 'r')

axes[1,2].scatter(x, y)

axes[1,2].set_title("m = 9")
#Type your code here -

"""

Use PolynomialFeatures() to generate a feature matrix

"""



from sklearn.preprocessing import PolynomialFeatures



X=[2, -1, 3]

X_reshape = np.array(X).reshape(len(X), 1) # Transpose

PolynomialFeatures(degree=2, include_bias=False).fit_transform(X_reshape)
"""

Use sklearn to generate the feature matrix of a quadratic polynomial

"""



from sklearn.preprocessing import PolynomialFeatures



x = np.array(x).reshape(len(x), 1) # Transpose

y = np.array(y).reshape(len(y), 1)



poly_features = PolynomialFeatures(degree=2, include_bias=False)

poly_x = poly_features.fit_transform(x)



poly_x
#Type your code here-

"""

Convert to linear regression predictions

"""



from sklearn.linear_model import LinearRegression



# Define linear regression model



model = LinearRegression()

model.fit(poly_x, y) # Training



# Obtain the linear regression parameters



model.intercept_, model.coef_
"""Plot the fitted figure

"""



x_temp = np.array(x_temp).reshape(len(x_temp),1)

poly_x_temp = poly_features.fit_transform(x_temp)



plt.plot(x_temp, model.predict(poly_x_temp), 'r')

plt.scatter(x, y)
"""

Load the dataset and preview

"""



import pandas as pd



df = pd.read_csv("../input/week4data/vaccine.csv", header=0)

df
#Type your code here-

"""

Plot

"""



#Define x and y

x = df['Year']

y = df['Values']



# Plot

plt.plot(x, y, 'r')

plt.scatter(x, y)
#Type your code here-

"""

Data partition

"""



#Divide dataframe into training set and testing set

train_df = df[:int(len(df)*0.7)] 

test_df = df[int(len(df)*0.7):]



# Define x and y for training and testing, respectively

train_x = train_df['Year'].values

train_y = train_df['Values'].values



test_x = test_df['Year'].values

test_y = test_df['Values'].values
#Type your code here-

"""

Linear regression predictions

"""



#Implement linear regression model

model = LinearRegression()

model.fit(train_x.reshape(len(train_x),1), train_y.reshape(len(train_y),1))

results = model.predict(test_x.reshape(len(test_x),1))

results # testing result
#Type your code here-

"""

Errors of linear regression

"""



from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



print("MAE of linear regression: ", mean_absolute_error(test_y, results.flatten()))

print("MSE of linear regression: ", mean_squared_error(test_y, results.flatten()))
#Type your code here-

"""

Quadratic polynomial predictions

"""



#Generate the feature matrix

poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)

poly_train_x_2 = poly_features_2.fit_transform(train_x.reshape(len(train_x),1))

poly_test_x_2 = poly_features_2.fit_transform(test_x.reshape(len(test_x),1))



# Training and predicting

model = LinearRegression()

model.fit(poly_train_x_2, train_y.reshape(len(train_x),1)) # Train the model



results_2 = model.predict(poly_test_x_2) # Prediction



results_2.flatten() # Print after flattening
#Type your code here-

"""

Errors of quadratic polynomial regression

"""



print("MAE of quadratic polynomial regression: ", mean_absolute_error(test_y, results_2.flatten()))

print("MSE of quadratic polynomial regression: ", mean_squared_error(test_y, results_2.flatten()))
#Type your code here-

"""

Polynomial regression predictions with higher degrees

"""



from sklearn.pipeline import make_pipeline



train_x = train_x.reshape(len(train_x),1)

test_x = test_x.reshape(len(test_x),1)

train_y = train_y.reshape(len(train_y),1)



for m in [3, 4, 5]:

    model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())

    model.fit(train_x, train_y)

    pre_y = model.predict(test_x)

    print("{}-degree polynomial regression_MAE: ".format(m), mean_absolute_error(test_y, pre_y.flatten()))

    print("{}-degree polynomial regression_MSE: ".format(m), mean_squared_error(test_y, pre_y.flatten()))

    print("---")
#Type your code here-

"""

Calculate MSE results of m-degree polynomial regression and plot

"""



mse = [] # Save MSE for different degrees

m = 1 # Start from 1-degree

m_max = 10 # Set the highest degree to be tested

while m <= m_max:

    model = make_pipeline(PolynomialFeatures(m, include_bias=False), LinearRegression())

    model.fit(train_x, train_y) # Train

    pre_y = model.predict(test_x) # Test

    mse.append(mean_squared_error(test_y, pre_y.flatten())) # Calculate MSE

    m = m + 1



print("MSE results: ", mse)

# Plot

plt.plot([i for i in range(1, m_max + 1)], mse, 'r')

plt.scatter([i for i in range(1, m_max + 1)], mse)



#Descriptions



plt.title("MSE of m degree of polynomial regression")

plt.xlabel("m")

plt.ylabel("MSE")
#Type your code here-
