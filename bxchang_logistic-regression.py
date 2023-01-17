import numpy as np

import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.show()

print('Data set size:', x_total.shape[0])
from sklearn import linear_model



lr_clf = linear_model.LogisticRegression()

lr_clf.fit(x_total, y_total)

print(lr_clf.coef_[0])

print(lr_clf.intercept_)



y_pred = lr_clf.predict(x_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (lr_clf.coef_[0][0] * plot_x + lr_clf.intercept_[0]) / lr_clf.coef_[0][1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
# 1. finish function my_logistic_regression;

# 2. draw a training curve (the x-axis represents the number of training iterations, and the y-axis represents the training loss for each round);

# 3. draw a pic to show the result of logistic regression (just like the pic in section 2);



n_iterations = 2000

learning_rate = 0.1

loss_list = []

def my_logistic_regression(x_total, y_total):

    # TODO

    LR = 0.01

    theta = np.ones((3, 1)) * 0.1

    x_total = np.concatenate((x_total, np.ones((100, 1))), axis=1)

    y_total = np.expand_dims(y_total, axis=1)

    g = lambda x, theta: (np.exp(theta.T.dot(x_total.T)) / (1 + np.exp(theta.T.dot(x_total.T)))).T

    loss = lambda x, y, theta: (-y.T.dot(np.log(g(x, theta))) - (1 - y).T.dot(np.log(1 - g(x, theta)))) / len(y)

    losses = np.zeros(500)

    for i in range(500):

        theta = theta - LR * x_total.T.dot(g(x_total, theta) - y_total)

        losses[i] = loss(x_total, y_total, theta)

    y_total = g(x_total, theta)

    y_total = np.round(y_total).astype(int).squeeze()

    return y_total, theta, losses



y_pred, theta, losses = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (theta[0] * plot_x + theta[2]) / theta[1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()

plt.figure(2)

plt.plot(losses)

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()