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

    # TODO

    global n_iterations

    global learning_rate

    global loss_list

    

    y_total_copy = y_total.copy()

    y_total_copy[y_total_copy==0] = -1

    

    w0 = np.random.normal(loc=0.0, scale=1.0)

    w1 = np.random.normal(loc=0.0, scale=1.0)

    # w = np.random.normal(loc=0.0, scale=1.0, size=2)

    b = np.random.normal(loc=0.0, scale=1.0)



    for i in range(n_iterations):

        y = x_total[:,0]*w0 + x_total[:,1]*w1 + b

        loss = 0.5 * np.mean((y-y_total_copy)**2)

        dw0 = np.mean((y-y_total_copy)*x_total[:, 0])

        dw1 = np.mean((y-y_total_copy)*x_total[:, 1])

        db  = np.mean(y-y_total_copy)



        w0 = w0 - learning_rate * dw0

        w1 = w1 - learning_rate * dw1

        b  = b  - learning_rate * db



        loss_list.append(loss)



    y_pred = x_total[:, 0]*w0 + x_total[:, 1]*w1 + b

    y_pred[y_pred>0] = 1

    y_pred[y_pred<0] = 0





    return y_pred, (w0, w1, b)



y_pred, (w0, w1, b) = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())
plt.plot(np.linspace(0, n_iterations-1, n_iterations), loss_list, c='g')

plt.title('Loss')

plt.show()
plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (w0 * plot_x + b) / w1

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()