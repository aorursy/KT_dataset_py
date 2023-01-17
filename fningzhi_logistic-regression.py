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

theta = np.zeros([1,3])



def sigmoid(x):

    return 1.0/(1.0+np.exp(-x))



def my_logistic_regression(x_total, y_total):

    # TODO

    global theta

    cons = np.ones((x_total.shape[0], 1))

    x_total = np.concatenate(( x_total,cons), axis=1)#行拼接

    y_copy=y_total.reshape(100,1)#转为可进行矩阵乘法的形式

    #print("y_copy",y_copy)

    #print("y_total",y_total)

    

    for i in range(n_iterations):

        y=np.dot(x_total,theta.T)

        y_train=sigmoid(y)

        #print("y_total-y_train",i,y_total-y_train)

        gradient = np.dot(x_total.T, (y_copy - y_train))

        #print("gradient",i,gradient)

        theta=theta+(gradient.T)*learning_rate

        #print("theta",i,theta)

        y_train=y_train.reshape(100,)#转为可进行正常运算的形式

        loss=-y_total*np.log(y_train) - (1-y_total)*np.log(1-y_train)

        loss_list.append(np.mean(loss))

    #阈值设为0.5

    y_train[y_train>0.5] = 1

    y_train[y_train<0.5] = 0



    return y_train



y_pred = my_logistic_regression(x_total, y_total)

print("y_pred",y_pred)

print("theta",theta)

print('accuracy:',(y_pred == y_total).mean())

#result of logistic regression

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (theta[0][1] * plot_x + theta[0][2]) / theta[0][0]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
#training curve

plt.plot(np.linspace(0, n_iterations-1, n_iterations), loss_list, c='g')

plt.title('Loss')

plt.show()