import numpy as np

import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')



pos_index = np.where(y_total == 1)

neg_index = np.where(y_total == 0)

# in set 'x total', the first column(0) is the x-axis and the second column(1) is the y-axis

# both range from -1 to 1

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.show()

print('Data set size:', x_total.shape[0])
from sklearn import linear_model



lr_clf = linear_model.LogisticRegression()

lr_clf.fit(x_total, y_total)

print(lr_clf.coef_[0])

# w1 w2

print(lr_clf.intercept_)

# w0



y_pred = lr_clf.predict(x_total)

print('accuracy:',(y_pred == y_total).mean())



plot_x = np.linspace(-1.0, 1.0, 100) # the straight line: w1x1 + w2x2 + w0 = 0

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

    w = np.zeros([1, x_total.shape[1]+1]) # w0 w1 w2，coefficient

    ones = np.ones([x_total.shape[0], 1])

    x_0 = np.column_stack((ones, x_total)) #add one column to x_total

    y_total.resize((x_total.shape[0],1)) # 1D → 2D

    

    loss_list, w = my_gradient_descent(x_0, y_total, w, learning_rate, n_iterations)

    draw_loss(loss_list, n_iterations)

    draw_scatter(x_total, w)

    

    y_pred = np.dot(x_0, w.T)

    for i in range(y_pred.shape[0]):

        if y_pred[i][0] > 0.5:

            y_pred[i][0] = 1

        else:

            y_pred[i][0] = 0 # classify

    return y_pred







def my_gradient_descent(x, y, w, learning_rate, n_iterations):

    iter = 0

    gradient = np.zeros([1, w.shape[1]])

    Loss = [cal_loss(x, y, w)]

    while True:

        gradient = cal_gradient(x, y, w)

        w = w - learning_rate * gradient

        iter += 1

        Loss.append(cal_loss(x, y, w))

        if iter == n_iterations - 1:

            break;

    return Loss, w





# calculate gradient

def cal_gradient(x, y, w):

    gradient = np.zeros(w.shape)

    z = np.dot(x, w.T) # z = wTx

    mid = (np.subtract(sigmoid(z), y)).ravel() # referred to some textbooks, it seems like flatten the original array?

    for i in range(w.shape[1]):

        gradient[0, i] = np.sum(np.multiply(mid, (x[:,i]))) / (x.shape[0])

    return gradient







def sigmoid(x):

    y = 1.0/(1.0 + np.exp(-x))

    return y





def cal_loss(x, y, w):

    z = np.dot(x, w.T)

    H = sigmoid(z)

    L0 = np.sum(np.multiply(-y,np.log(H)) - np.multiply(1-y,np.log(1-H)))  

    L = L0 / x.shape[0]    

    return L



    



def draw_loss(loss, n_iterations):

    plt.plot(range(n_iterations),loss)

    plt.show()

    





def draw_scatter(x, w):    

    plot_x = np.linspace(-1.0, 1.0, 100) 

    plot_y = - (w[0][1] * plot_x + w[0][0]) / w[0][2]

    plt.scatter(x[pos_index, 0], x[pos_index, 1], marker='o', c='r')

    plt.scatter(x[neg_index, 0], x[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())
