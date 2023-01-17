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



def sigmoid(x):

    # Activation function used to map any real value between 0 and 1

    return 1 / (1 + np.exp(-x))



def my_logistic_regression(x_total, y_total):

    # TODO

    constant_x = np.ones((100,1))

    x_total = np.c_[constant_x,x_total]

    x_total = np.mat(x_total)

    y_total = np.mat(y_total).transpose()

    

    # a = number of training examples

    # b = number of features

    a,b = x_total.shape

    weights = np.ones((b,1))

    

    j_cost = np.zeros((a,1))

    j_cost_sum = np.zeros((n_iterations,1))

    

    for i in range(n_iterations):

        hx = sigmoid(x_total * weights)

        # compute cost for given parameters

        j_cost = -np.multiply(y_total,np.log10(hx)) - np.multiply((1-y_total),np.log10(1-hx))

        j_cost_sum[i] = sum(j_cost)

        # calculate error

        error = y_total - hx

        # update weights

        weights = weights + learning_rate * x_total.transpose() * error

    

    plt.plot(range(1, n_iterations + 1), j_cost_sum)

    plt.xlabel('Training Iterations')

    plt.ylabel('Training Loss')

    plt.title('Logistic Regression Training Curve')

    plt.show()

    

    return weights



def accuracy(weights, x_total, y_total, prob_threshold=0.5):

    constant_x = np.ones((100,1))

    x_total = np.c_[constant_x,x_total]

    a,b = x_total.shape

    count = 0

    

    for i in range(a):

        predict = sigmoid(x_total[i, :] * weights)[0, 0] >= prob_threshold

        if predict == bool(y_total[i]):

            count += 1

    acc = float(count) / a

    

    return acc



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',accuracy(y_pred, x_total, y_total))



x_values = np.linspace(-1.0, 1.0, 100)

y_values = (-y_pred[0,0] - y_pred[1,0] * x_values) / y_pred[2,0]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(x_values, y_values, c='g')

plt.title('Logistic Regression - Gradient Descent')

plt.show()