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



def my_logistic_regression(x_total, y_total, n, lr):

    # TODO

    loss_list = []

    w1 = 0

    b = 0

    for i in range(n):

        z = w1 * x_total[:,0] + x_total[:,1] + b

        y_pred = 1 / (1 + np.exp(0-z))

        dw = np.sum((y_pred - y_total) * x_total[:,0]) / x_total.shape[0]

        db = np.sum(y_pred - y_total) / x_total.shape[0]

        w1 = w1 - lr * dw

        b = b - lr*b

        loss = -np.sum(y_total*np.log(y_pred) + (1-y_total)*np.log(1-y_pred)) / x_total.shape[0]

        loss_list.append(loss)

    y_pred = (y_pred > 0.5).astype(np.int16)

    return y_pred, w1, b, loss_list



y_pred,w,b,loss_list = my_logistic_regression(x_total, y_total, n_iterations, learning_rate)

it = np.arange(0,n_iterations,1)

plt.plot(it, loss_list)

plt.show()

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (w * plot_x + b)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()

print('accuracy:',(y_pred == y_total).mean())