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





def sigmoid(z):

    return 1/(1+np.exp(-z))



def my_logistic_regression(x_total, y_total):

    

    x_total = np.column_stack((x_total, np.ones(x_total.shape[0])))

    theta = np.ones(x_total.shape[1])

    for i in range(n_iterations):

        h_x = sigmoid(np.dot(x_total, theta.T))

        

        grad = np.dot(x_total.T, (h_x - y_total))

        

        theta = theta - learning_rate * grad

      

        loss = (y_total.T).dot(x_total.dot(theta.T))-sum(np.log(1 + np.exp(x_total.dot(theta.T))))

        loss_list.append(loss)

        

    y_pred = np.dot(x_total, theta.T) > 0.5

    

    return y_pred, theta, loss_list



y_pred, theta, loss_list = my_logistic_regression(x_total, y_total)



print('accuracy:',(y_pred == y_total).mean())



##draw func



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (theta[2] + theta[0] * plot_x) / theta[1]



figure1 = plt.figure()

plt.title('training curve')

plt.xlabel('iterations')

plt.ylabel('loss')



plt.plot(np.arange(1, n_iterations + 1), loss_list)



fugure2 = plt.figure()

plt.title('my_logistic_regression')

plt.ylim(-1, 1)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')





plt.show()
