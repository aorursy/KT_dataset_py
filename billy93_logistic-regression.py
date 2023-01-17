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



import matplotlib.pyplot as plt



n_iterations = 2000

learning_rate = 0.02



class Linear_Regression():

    def __init__(self, n_iterations, learning_rate):

        self.n_iter = n_iterations

        self.lr = learning_rate

        self.loss_list = []

    

    def fit(self, x_total, y_total):

        [x_row, x_col] = x_total.shape

        self.W = np.ones([x_col, 1])

        self.b = [1]

        y_total = np.resize(y_total, [len(y_total), 1])

        

        for step in range(self.n_iter):

            pred = 1 / (1 + np.exp(-(np.dot(x_total, self.W) + self.b)))

            self.W -= self.lr * x_total.T.dot(pred-y_total) / x_row

            self.b -= sum(self.lr * (pred-y_total)) / x_row

            loss = sum((pred - y_total)**2) / (2*x_row)

            self.loss_list.append(loss)



    def predict(self,x):

        pred = 1 / (1 + np.exp(-(np.dot(x_total, self.W) + self.b)))

        for i in range(pred.shape[0]):

            if pred[i][0] >= 0.5:

                pred[i][0] = 1

            else:

                pred[i][0] = 0

        return np.resize(pred, [100])

    

    def fig(self):

        iter = list(range(self.n_iter))

        plt.scatter(iter, self.loss_list)

        plt.plot(iter, self.loss_list)

        plt.xlabel("Iterations")

        plt.ylabel("Loss")

        plt.show()

    

    def result(self, x_total, y_total, pred):

        y_total = np.resize(y_total, [100, 1])

        plot_x = np.linspace(-1.0, 1.0, 100)

        plot_x = np.resize(plot_x, [100, 1])

        plot_y = - (self.W[0][0] * plot_x + self.b[0]) / self.W[1][0]

        

        print(self.W)

        print(self.b)

        

        plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

        plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

        plt.plot(plot_x, plot_y, c='g')

        plt.show()



clf = Linear_Regression(n_iterations, learning_rate)

clf.fit(x_total, y_total)

pred = clf.predict(x_total)

clf.fig()

clf.result(x_total, y_total, pred)

print('accuracy:',(pred == y_total).mean())