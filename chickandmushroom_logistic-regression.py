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

    result = 1.0/(1+np.exp(-x))

    return result



def my_logistic_regression(x_total, y_total,loss_list,n_iterations = 2000,learning_rate = 0.1):

    N = x_total.shape[0]

    w = np.ones((3,1))

    tmp = np.ones((N,1))

    X = np.hstack((tmp,x_total))

    Y = np.mat(y_total).T

    pred = np.ones((N,1))

    for i in range(n_iterations):

        pred = sigmoid(np.dot(X,w))

        grad = np.dot(np.mat(X).T,(pred-Y))

        

        w = w-learning_rate*grad

        loss = -np.dot(y_total,np.log2(pred))-np.dot((1-y_total),np.log2(1-pred)) #CrossEntropy

        loss_sum = np.sum(loss)

        loss_list.append(loss_sum)

    

    y_pred = np.round(pred)



    return y_pred,w



y_pred,w = my_logistic_regression(x_total, y_total,loss_list)



print('accuracy:',(y_pred == np.mat(y_total).T).mean())



plot_y=loss_list

plt.plot(plot_y, c='g')

plt.xlabel('iterations')

plt.ylabel('loss')

plt.show()


plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (w[1] * plot_x + w[0]) / w[2]

plot_y = np.mat(plot_y).T

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()