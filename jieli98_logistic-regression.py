import numpy as np

import matplotlib.pyplot as plt

from numpy import dot

import pandas as pd



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')

print(y_total)



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



J = pd.Series(np.arange(n_iterations, dtype = float))

def sigmoid(x):

    return 1.0/(1+np.exp(-x))



def my_logistic_regression(x_total, y_total):

    x_total=np.insert(x_total, 0, 1, axis=1) 

    m,n=np.shape(x_total)

    theta=np.zeros((n,1))

    y=y_total.copy()

    y=y.reshape(m,1)

    

    for i in range(n_iterations):

        pred=[]

        h=sigmoid(np.dot(x_total,theta))

        j=-(1/100.)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))

        J[i] = -(1/100.)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))

        loss=y-h   

        theta = theta + learning_rate *  np.dot(x_total.T, loss)

        loss_list.append(j)

    

    prob=sigmoid(np.dot(x_total,theta))



    for i in range(m):

        if prob[i]>0.5:

            y[i]=1

        else:

            y[i]=0

    y=y.reshape(m,)  

   

    return y,theta







y_pred,theta = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())



plt.figure(figsize=(20,8))

plt.subplot(121)

plt.plot(loss_list)

plt.subplot(122)

plot_x1 = np.linspace(-1.0, 1.0, 100)

plot_y = - (theta[1] * plot_x1 +theta[0]) / theta[2]



plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x1, plot_y, c='g')

plt.show()








