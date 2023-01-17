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

x1_total = lines[:, 1:3].astype('float')

y1_total = lines[:,3].astype('float')

x1_total = np.insert(x1_total, 0, 1, axis=1)

def predict(weights, x_total):

    probability = 1 / (1 + np.exp(-(x_total*weights)))

    return [1 if x >= 0.5 else 0 for x in probability]

def my_logistic_regression(x_total, y_total):

    x_total = np.mat(x_total)

    y1_total = np.mat(y_total).transpose()

    m, n = np.shape(x_total)

    weights = np.ones((n, 1))

    costs = np.ones(n_iterations)

    for i in range(n_iterations):

        h = 1 / (1 + np.exp(-(x_total * weights)))

        weights = weights + learning_rate  *x_total.transpose() * (y1_total- h) 

        costs[i] = np.mean((-y_total) * np.log(1 / (1 + np.exp(-(x_total @ weights)))) - (1 - y_total)*np.log(1 - 1 / (1 + np.exp(-(x_total @ weights)))))

    count = [[i] for i in range(1,2001)]

    plt.plot(count,costs)

    plt.title('Train Loss : Iterations',color='red')

    plt.show()

    return weights,costs

[weights,costs] = my_logistic_regression(x1_total,y1_total)

y_pred = predict(weights,x1_total)

def draw_figure(weights, x_total, y_total):

    x = np.linspace(-1.0, 1.0, 100)

    y = (-weights[0,0] - weights[1,0] * x) / weights[2,0]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(x, y, c = 'g')

    plt.show()

draw_figure(weights, x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())