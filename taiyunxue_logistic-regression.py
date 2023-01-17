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

import numpy as np

n_iterations = 2000

learning_rate = 0.1

loss_list = []



def sigmoid(inX):

    return 1.0 / (1 + np.exp(-inX))



def my_logistic_regression(x_total, y_total):

    # TODO

    

    y_total = y_total.reshape((100, 1))

    numSamples, numFeatures = np.shape(x_total)

    weights = np.ones((numFeatures, 1))

    b=0

    #predict过程

    for k in range(n_iterations):

            output = sigmoid(x_total.dot(weights) + b)

            error = y_total - output

            loss = -np.multiply(y_total, np.log(output)) - np.multiply((1 - y_total), np.log(1 - output))

            loss = np.sum(loss) / numFeatures

            loss_list.append(loss)

            weights = weights + learning_rate * x_total.T.dot(error)/numSamples

            b = b + learning_rate * np.sum(error, axis=0, keepdims=True) / numSamples

    #矩阵值转换为0，1

    y_total=sigmoid(x_total.dot(weights) + b)

    y_total[y_total >= 0.5] = 1

    y_total[y_total < 0.5] = 0

    #loss可视化

    x0 = range(n_iterations)

    plt.plot(x0, loss_list)

    plt.show()

    #LG可视化

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = -weights[0][0] / weights[1][0] * plot_x - b / weights[1][0]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y.T, c='g')

    plt.show()

    return y_total.T



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())
