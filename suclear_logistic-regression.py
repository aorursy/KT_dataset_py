import numpy as np

import matplotlib.pyplot as plt

import copy



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.show()

print('Data set size:', x_total.shape[0])

y_orig=copy.deepcopy(y_total)
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

    '''

    由于我没有找到办法让function不改变y_total，故在开始的时候将它copy为y_orig以用于最后accuracy的计算，抱歉

    '''

    n = x_total.shape[0]

    ext = np.ones((n,1))

    x_totalmat = np.mat(x_total)

    x_mat = np.concatenate([ext,x_totalmat],1)

    y_mat = np.mat(y_total)

    w = np.ones((3,1))

    for i in range(n_iterations):

        sig = 1 / (1 + np.exp( - x_mat *  w))

        loss = (y_mat.transpose() - sig)

        loss_list.append(abs(loss.mean()))

        w = w + learning_rate * x_mat.transpose() * loss

    for j in range(n):

        if - (w[0] + w[1] * x_total[j,0]) / w[2] > x_total[j,1]:

            y_total[j] = 0

        else:

            y_total[j] = 1

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (w[1,0] * plot_x + w[0,0]) / w[2,0]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()

    plot_x = np.linspace(1, 2000, 2000)

    plt.plot(plot_x, loss_list)

    plt.show()

    return y_total

y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_orig).mean())