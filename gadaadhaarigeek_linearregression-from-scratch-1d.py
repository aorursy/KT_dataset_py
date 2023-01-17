# Generating 1D data 

import numpy as np 

import matplotlib.pyplot as plt



# Number of data points

N = 100



X = np.random.uniform(low=0, high=100, size=N)



# Making y = 2x + 1 + some gaussian or normal noise (assumption of linear regression itself)

Y = 2 * X + 1 + np.random.normal(scale=10, size=N)



# Plotting the data to see if it looks linear

plt.scatter(X, Y, edgecolors='black', color="red")

plt.show()
# Applying linear regression

# After doing the OLS minimizations we get below eqautions for w and b

denominator = X.dot(X) - X.mean() * X.sum()

w = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator

b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator
Yhat = w * X + b
plt.scatter(X, Y, edgecolors='black')

plt.plot(X, Yhat, color="red")

plt.show()
# It seems our model (line) has done pretty well in identifying the relationship. Of course we made up the data. 

# Let's look at the weights 

print("w: ", w, " and b: ", b)
# w which is the slope of line is close to 2 (as in our original data) and bias term is having 1 and the gaussian 

# noise which we added
# calcualting MSE 

mse = (Yhat - Y).dot(Yhat - Y)/N

print("MSE: ", mse)
# MSE or even RMSE doesn't really tell about the model. So we cxalculate R^2 which tells how good our model fit the 

# data

ss_res = Y - Yhat

ss_tot = Y - Y.mean()

r2 = 1 - ss_res.dot(ss_res)/ss_tot.dot(ss_tot)

print("The r-squared is: ", r2)
# This value of r-suared is pretty good, as it can be maximum of 1. This means our data is modeled properly by our 

# model