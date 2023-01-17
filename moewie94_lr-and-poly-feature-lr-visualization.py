import numpy as np

from sklearn.datasets import make_moons

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
# generate data

# generate training set

data_train, label_train = make_moons(1000, noise=0.15)

# generate testing set

data_test, label_test = make_moons(200, noise=0.15)
data_train
# plot data

# plot training samples

plt.scatter(data_train[:,0], data_train[:,1], c=label_train, cmap='winter')

# plot testing samples

plt.scatter(data_test[:,0], data_test[:,1], c=label_test, cmap='summer')
# call logistic regression model

model = LogisticRegression()

# fit model to data

model.fit(data_train, label_train)

# show model parameters

print('W:', model.coef_, 'b:', model.intercept_)
# plot data

# plot training samples

plt.scatter(data_train[:,0], data_train[:,1], c=label_train, cmap='winter')

# plot testing samples

plt.scatter(data_test[:,0], data_test[:,1], c=label_test, cmap='summer')

# plot decision boundary

plt.plot(data_train[:,0], (0-model.intercept_ - model.coef_[0, 0]*data_train[:, 0])/model.coef_[0, 1], 'r')
# predict with trained model

predict_test = model.predict(data_test)

# calculate accuracy score

accuracy_test = accuracy_score(label_test, predict_test)

# print accuracy level

print(accuracy_test)
# call polynomial feature transformer

poly_tfm = PolynomialFeatures(degree=5)

# transform features

data_train_poly = poly_tfm.fit_transform(data_train)

# call logistic regression model

model = LogisticRegression()

# fit model on transformed train data

model.fit(data_train_poly, label_train)
# transform features

data_test_poly = poly_tfm.fit_transform(data_test)

# predict with trained model

predict_test = model.predict(data_test_poly)

# calculate accuracy score

accuracy_test = accuracy_score(label_test, predict_test)

# print accuracy level

print(accuracy_test)
# Create a meshgrid (like what we did with the linear transformation matrix)

# x coordinates (-1.5 to 2.5, space 0.01)

xm = np.arange(-1.5, 2.5, 0.01)

# y coordinates (-1 to 2, space 0.01)

ym = np.arange(-1, 2, 0.01)

# 2d mesh grid (or we can call it matrix of coordinates)

xx, yy = np.meshgrid(xm, ym)

# flatten to reshape to (n_coordinate, 2) to match model input

xx_flatten = xx.reshape(-1, 1)

yy_flatten = yy.reshape(-1, 1)

coordinates = np.concatenate((xx_flatten, yy_flatten), axis = 1)
# transform coordinate (just like normal data samples)

mesh_poly = poly_tfm.fit_transform(coordinates)

# predict with trained model

predict_mesh = model.predict(mesh_poly)

predict_mesh = predict_mesh.reshape(xx.shape)
# plot data

# plot training samples

plt.scatter(data_train[:,0], data_train[:,1], c=label_train, cmap='winter')

# plot testing samples

plt.scatter(data_test[:,0], data_test[:,1], c=label_test, cmap='summer')

plt.contourf(xx, yy, predict_mesh, 2, cmap='spring', alpha = .1)
# plot data

# plot testing samples

plt.scatter(data_test[:,0], data_test[:,1], c=label_test, cmap='jet')

plt.contourf(xx, yy, predict_mesh, 2, cmap='winter', alpha = .2)