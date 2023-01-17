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

    q = np.ones((100,1))

    x_total = np.c_[q,x_total]

    w = np.ones((3,1))

    y_total = y_total.reshape(1,100)

    for k in range(n_iterations):

        output = 1.0/(1+np.exp(-np.dot(x_total,w)))

        error = output - y_total.transpose() 

        w = w - learning_rate*np.dot(x_total.transpose(),error)

        L = (-np.dot(y_total,np.log(output)) -np.dot ((1 - y_total),np.log(1 - output))).squeeze()/100

        loss_list.append(L)

    y_total = 1.0/(1+np.exp(-1*np.dot(x_total,w)))

    y_total = np.round(y_total).squeeze()

    return y_total,w,loss_list



y_pred,weights,loss_list = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())





plt.figure(figsize=(10, 6))

plt.plot([i for i in range(len(loss_list))], loss_list)

plt.xlabel("Number of iterations")

plt.ylabel("Loss function")

plt.show()





for i in range(100):

    if int(y_total[i]) == 0:

        plt.plot(x_total[i,0], x_total[i,1], marker ='x', c='b')

    elif int(y_total[i]) == 1:

        plt.plot(x_total[i,0], x_total[i,1], marker ='o', c='r')

min_x = min(x_total[:, 0])

max_x = max(x_total[:, 0])

y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]

y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]

plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')

plt.show()