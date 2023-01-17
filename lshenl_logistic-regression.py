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

learning_rate = 0.01

loss_list = []



def sigmoid(x):

    p = np.exp(x)/(1+np.exp(x))

    return p



def my_logistic_regression(x_total, y_total):

    x_total = np.c_[np.ones(x_total.shape[0]), x_total]

    theta = np.ones(x_total.shape[1])

    for i in range(n_iterations):

        h = sigmoid(x_total.dot(theta.T))

        grad = (x_total.T).dot(y_total - h)

        theta = theta + (learning_rate * grad).T

        

        cost = (y_total.T).dot(x_total.dot(theta.T))-sum(np.log(1 + np.exp(x_total.dot(theta.T))))

        loss_list.append(cost)

        

    y_pred = np.int64(x_total.dot(theta.T) > 0)

    return y_pred, theta, loss_list



y_pred, theta, loss_list = my_logistic_regression(x_total, y_total)



print(theta)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (theta[0] + theta[1] * plot_x) / theta[2]



plt.figure(1)

plt.plot(np.arange(1, n_iterations + 1), loss_list)

plt.xlabel('Iterations')

plt.ylabel('Loss')

plt.title('Training Curve')

plt.figure(2)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.title('The Result of Logistic Regression')

plt.show()