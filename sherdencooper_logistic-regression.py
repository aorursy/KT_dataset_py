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

    a = 1/(1+np.exp(-z))    

    return a



def initial_para():

    w = np.zeros((2,1))

    b = 0

    return w, b



def my_logistic_regression(x_total, y_total):

    w,b = initial_para()    #initialize parameters

    num =100   #count the number of instances

    for i in range(n_iterations):

        #forward propogation

        A = sigmoid(np.dot(x_total, w)+b)

        cost = -(np.sum(y_total*np.squeeze(np.log(A))+(1-y_total)*np.squeeze(np.log(1-A))))/num

        loss_list.append(cost)       #record the loss

        #backward  propogation

        dZ = A - y_total[:, np.newaxis]

        dw = (np.dot(x_total.T,dZ))/num

        db = (np.sum(dZ))/num

        w = w - learning_rate*dw

        b = b - learning_rate*db

    #prediction

    output = np.squeeze(A)

    for i in range(num):

        if output[i]>=0.5:

            output[i]=1

        else:

            output[i]=0

    y_total = output

    plt.figure(figsize=(15,5))

    ax1 = plt.subplot(1,2,1)

    ax2 = plt.subplot(1,2,2)

    plt.sca(ax1)

    x= np.linspace(1,n_iterations,n_iterations, dtype=int)

    plt.plot(x,loss_list)

    plt.sca(ax2)

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (w[0,0] * plot_x + b) / w[1,0]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()

    

    return y_total



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())