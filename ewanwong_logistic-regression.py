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

        # 修改数据规模

    y_total = y_total.reshape((100, 1))

    m = x_total.shape[0]

    n = x_total.shape[1]

    # 初始化weight和bias

    w = np.zeros(shape=(n, 1))

    b = 0

    # 学习

    for i in range(n_iterations):

        z = np.dot(x_total, w) + b

        a = 1 / (1 + np.exp(-z))

        loss = -np.multiply(y_total, np.log(a)) - np.multiply((1 - y_total), np.log(1 - a))

        loss = np.sum(loss) / m

        loss_list.append(loss)

        dw = np.dot(x_total.T, (a - y_total)) / m

        db = np.sum((a - y_total), axis=0, keepdims=True) / m

        w -= learning_rate * dw

        b -= learning_rate * db

    # 绘制cost funktion曲线

    it = range(n_iterations)

    plt.plot(it, loss_list)

    plt.show()

    # 计算training set的预测

    z_pred = np.dot(x_total, w) + b

    a_pred = 1 / (1 + np.exp(-z_pred))

    a_pred[a_pred >= 0.5] = 1

    a_pred[a_pred < 0.5] = 0

    # 数据可视化

    plt_x = np.linspace(-1.0, 1.0, 100)

    plt_y = (-w[0][0] / w[1][0] * plt_x - b / w[1][0])

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plt_x, plt_y.T, c='g')

    plt.show()

    return a_pred.T





y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())