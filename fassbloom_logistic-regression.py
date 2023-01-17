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

    y_total = y_total.reshape((100, -1))

    weight = np.zeros(shape=(x_total.shape[1], 1))

    bias = 0

    for i in range(n_iterations):

        a = 1 / (1 + np.exp(-(np.dot(x_total, weight) + bias)))

        dw = np.dot(x_total.T, (a - y_total)) / x_total.shape[0]

        db = np.sum((a - y_total), axis=0, keepdims=True) / x_total.shape[0]

        weight -= learning_rate * dw

        bias -= learning_rate * db

        loss = -np.multiply(y_total, np.log(a)) - np.multiply((1 - y_total), np.log(1 - a))

        loss = np.sum(loss) / x_total.shape[0]

        loss_list.append(loss)

    

    a_pred = 1 / (1 + np.exp(-(np.dot(x_total, weight) + bias)))

    for i in range(100):

        if a_pred[i] >= 0.5:

            a_pred[i] = 1

        if a_pred[i] < 0.5:

            a_pred[i] = 0

   

    pltx = range(n_iterations)

    plt.plot(pltx, loss_list)

    plt.title("loss")

    plt.show()

    

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = (-weight[0][0] / weight[1][0] * plot_x - bias / weight[1][0])

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y.T, c='g')

    plt.title("my prediction")

    plt.show()

    return a_pred.T



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())