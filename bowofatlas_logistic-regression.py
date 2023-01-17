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

print(x_total.shape)
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



n_iterations = 6000

learning_rate = 0.1

loss_list = []

def my_logistic_regression(x_total, y_total):

    # TODO

    quantity = x_total.shape[0]

    w = np.ones((x_total.shape[1], 1)) # (2, 1)

    b = np.ones((1, 1)) * 0.5

    y_total = np.array(y_total).reshape((100, 1))

    for n in range(n_iterations):

        p = np.dot(x_total, w) + b

        p = 1.0 / (1 + np.exp(-p)) # (100, 1)

        loss = -np.sum((np.dot(np.log(p).T, y_total) + np.dot(np.log(1 - p).T, (1 - y_total)))) * (1 / quantity)

        loss = np.squeeze(loss)

        loss_list.append(loss)

        dw = np.dot(x_total.T, (p - y_total)) * (1 / quantity) # (2, 1)

        db = np.sum(p - y_total) * (1 / quantity)

        w = w - learning_rate * dw

        b = b - learning_rate * db

    plt.plot(range(n_iterations), loss_list)

    plt.xlabel("Number of Iterations")

    plt.ylabel("Loss")

    plt.show()

    

    y_pred = np.dot(x_total, w) + b

    y_pred = 1.0 / (1 + np.exp(-y_pred))

    y_pred = np.rint(y_pred)

    

    return y_pred, w, b





y_pred, w, b = my_logistic_regression(x_total, y_total)

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = np.squeeze(-(w[0][0] * plot_x + b) / w[1][0])

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()

acc = 0

for i in range(len(y_pred)):

    if y_pred[i] == y_total[i]:

        acc += 1

acc /= len(y_pred)

print("acc = ", acc)