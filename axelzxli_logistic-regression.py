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



def Sigmoid(z):

    res = 1.0 / (1.0 + np.exp(-1.0*z))

    return res



def Loss(h, y):

    return (-y * np.log(h) - (1 - y) * np.log(1 - h))



def my_logistic_regression(x_total, y_total):

    # TODO

    w = np.random.rand(2)

    b = np.ones(x_total.shape[0])



    for i in range(n_iterations):

        y_pred = Sigmoid(np.dot(x_total, w)+b)        

        slp_w = np.dot(x_total.T, (y_pred-y_total))/x_total.shape[0]

        slp_b = (y_pred - y_total).sum(axis = 0)/x_total.shape[0]

        w = w - slp_w*learning_rate

        b = b - slp_b*learning_rate

        loss = Loss(y_pred, y_total)

        loss_list.append(np.sum(loss))

    for j in range(x_total.shape[0]):

        if y_pred[j] > 0.5:

            y_pred[j] = 1

        else:

            y_pred[j] = 0

    print(w)

    print(b[0])

    return y_pred, w, b



y_pred, w, b = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())
plot_x = np.linspace(-1.0, 1.0, 100)

plot_x = plot_x.reshape((plot_x.shape[0],1))

plot_y = - (w[0] * plot_x + b[0]) / w[1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
plt.plot(loss_list)

plt.title('Loss')

plt.show