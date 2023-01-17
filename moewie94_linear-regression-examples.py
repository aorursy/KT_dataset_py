import numpy as np

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
# Generate data

# training data

training_samples = 1000

x_train = np.random.rand(training_samples,1)

noise_train = np.random.rand(training_samples,1) * 0.6

y_train = 24 + 4*x_train + noise_train



# testing data

testing_samples = 100

x_test = np.random.rand(testing_samples,1)

noise_test = np.random.rand(testing_samples,1) * 0.6

y_test = 24 + 4*x_test + noise_test
# plot data

# plot training data

plt.scatter(x_train, y_train, alpha=0.2)

# plot testing data

plt.scatter(x_test, y_test)
# call linear regression model from sklearn

model = LinearRegression()

# fit model to data

model.fit(x_train, y_train)

# show model parameters 

print('Coef:', model.coef_, 'Intercept:', model.intercept_)
# predict on test data

y_prediction = model.predict(x_test)
# calculate mse loss

prediction_error = mean_squared_error(y_test, y_prediction)



# print error

print('error:', prediction_error)
# calculate mae loss

mse_error = mean_absolute_error(y_test, y_prediction)

# print error

print('error:', mse_error)
# plot training samples

plt.scatter(x_train, y_train, alpha=0.2, label='training samples')

# plot testing samples

plt.scatter(x_test, y_test, alpha=0.4, label='testing samples')

# plot predicted testing samples

plt.scatter(x_test, y_prediction, label='predicted testing samples')

# plot fitting line

plt.plot(x_test, y_prediction, 'r', label='fitting line')

# add legend

plt.legend()
# Generate data

# training data

training_samples = 1000

x_train = np.random.rand(training_samples,1)

noise_train = np.random.rand(training_samples,1) * 0.9

y_train = 24 + 4*x_train + noise_train



# testing data

testing_samples = 100

x_test = np.random.rand(testing_samples,1)

noise_test = np.random.rand(testing_samples,1) * 0.9

y_test = 24 + 4*x_test + noise_test
# plot data

# plot training data

plt.scatter(x_train, y_train, alpha=0.2)

# plot testing data

plt.scatter(x_test, y_test)
# call linear regression model from sklearn

model = LinearRegression()

# fit model to data

model.fit(x_train, y_train)

# show model parameters 

print('Coef:', model.coef_, 'Intercept:', model.intercept_)
# predict on test data

y_prediction = model.predict(x_test)
# calculate mse loss

prediction_error = mean_squared_error(y_test, y_prediction)



# print error

print('error:', prediction_error)
# Generate data

# training data

training_samples = 1000

x_train = np.random.rand(training_samples,1)

noise_train = np.random.rand(training_samples,1) * 0.6

y_train = 5 + 2*x_train + 4*(x_train**2) + 2*(x_train**3) - 7*(x_train**4) - 2*(x_train**5) + noise_train



# testing data

testing_samples = 100

x_test = np.random.rand(testing_samples,1)

noise_test = np.random.rand(testing_samples,1) * 0.6

y_test = 5 + 2*x_test + 4*(x_test**2) + 2*(x_test**3) - 7*(x_test**4) - 2*(x_test**5) + noise_test
# plot data

# plot training data

plt.scatter(x_train, y_train, alpha=0.2)

# plot testing data

plt.scatter(x_test, y_test)
# call linear regression model from sklearn

model = LinearRegression()

# fit model to data

model.fit(x_train, y_train)

# show model parameters 

print('Coef:', model.coef_, 'Intercept:', model.intercept_)
# predict on test data

y_prediction = model.predict(x_test)
# calculate mse loss

prediction_error = mean_squared_error(y_test, y_prediction)



# print error

print('error:', prediction_error)
# plot training samples

plt.scatter(x_train, y_train, alpha=0.2, label='training samples')

# plot testing samples

plt.scatter(x_test, y_test, alpha=0.4, label='testing samples')

# plot predicted testing samples

plt.scatter(x_test, y_prediction, label='predicted testing samples')

# plot fitting line

plt.plot(x_test, y_prediction, 'r', label='fitting line')

# add legend

plt.legend()
# call polynomail features

poly_transform = PolynomialFeatures(degree=5)

x_train_poly = poly_transform.fit_transform(x_train)[:,1:]



# call linear regression model from sklearn

model = LinearRegression()

# fit model to data

model.fit(x_train_poly, y_train)

# show model parameters 

print('Coef:', model.coef_, 'Intercept:', model.intercept_)
# transfrom test samples

x_test_poly = poly_transform.fit_transform(x_test)[:,1:]

# predict on test data

y_prediction = model.predict(x_test_poly)
# calculate mse loss

prediction_error = mean_squared_error(y_test, y_prediction)



# print error

print('error:', prediction_error)
# plot training samples

plt.scatter(x_train, y_train, alpha=0.2, label='training samples')

# plot testing samples

plt.scatter(x_test, y_test, alpha=0.4, label='testing samples')

# plot predicted testing samples

plt.scatter(x_test, y_prediction, label='predicted testing samples')

# # plot fitting line

# plt.plot(x_test, y_prediction, 'r', label='fitting line')

# sort x_test index

sorted_index = np.argsort(x_test, axis=0)[:,0]

# plot fitting line using sorted index

plt.plot(x_test[sorted_index], y_prediction[sorted_index], 'r', label='fitting line')

# add legend

plt.legend()