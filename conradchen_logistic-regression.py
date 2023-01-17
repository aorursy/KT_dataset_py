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

class MyLogisticRegression (object):

    def __init__(self):

        self.weight = None

        self.intercept = None



    def fit(self, data, target, alpha=0.1, max_iteration=500):

        self.loss_list = np.zeros(max_iteration)

        assert len(data.shape) == 2

        nfeatures = data.shape[1]

        #print('nfeature:', nfeatures)

        nsamples = data.shape[0]

        #init W and b

        self.weight = np.zeros(nfeatures)

        self.intercept = np.zeros(1)

        for i in range(max_iteration):

            errors = self.prob(data) - target

            for j in range(nfeatures):

                self.weight[j] = self.weight[j] - alpha * np.mean(errors * data[:,j])

            self.intercept = self.intercept - alpha * np.mean(errors)

            self.loss_list[i] = self.cost(data, target)

        return self.loss_list



    def cost(self, data, target):

        y_prob = self.prob(data)

        #y*log(h(x)) + (1-y)*log(1-h(x))

        inner = target * np.log(y_prob) + (1-target) * np.log(1-y_prob)

        return -np.mean(inner)



    def predict(self, data):

        prob = self.prob(data)

        predict = [int(p >= 0.5) for p in prob]

        return np.asarray(predict)



    def prob(self, data):

        lin = np.dot(data, self.weight) + self.intercept

        #sigmoid

        return 1 / (1+np.exp(-lin))

def my_logistic_regression(x_total, y_total):

    mlr = MyLogisticRegression()

    mlr.fit(x_total, y_total,alpha = learning_rate, max_iteration = n_iterations)

    loss_list = mlr.loss_list

    plt.figure()

    plt.plot(loss_list)

    plt.show()

    y_total = mlr.predict(x_total)



    plt.figure()

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (mlr.weight[0] * plot_x + mlr.intercept[0]) / mlr.weight[1]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()

    return y_total



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())