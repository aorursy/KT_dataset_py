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

def my_logistic_regression(x_total, y_total,n = 2000, learning_rate = 0.1):

    ## 参数初始化

    w0 = 0

    w1 = 0

    w2 = 0

    

    ## 迭代计算

    for k in range(n):

        p = w0+w1*x_total[:,0]+w2*x_total[:,1]

        sigmoid = 1/(1+np.exp(-p))

        ## 计算loss

        loss = -y_total*np.log(sigmoid)-(1-y_total)*np.log(1-sigmoid)

        loss_total = loss.mean()

        loss_list.append(loss_total)

        ## 梯度下降

        grad0 = sigmoid - y_total                           

        grad1 = (sigmoid - y_total)*x_total[:,0]

        grad2 = (sigmoid - y_total)*x_total[:,1]

        w0 = w0 - learning_rate*grad0.mean()

        w1 = w1 - learning_rate*grad1.mean()

        w2 = w2 - learning_rate*grad2.mean()

    

    y_pred1 = w0+w1*x_total[:,0]+w2*x_total[:,1]

    y_pred1[y_pred1>0] = 1

    y_pred1[y_pred1<=0] = 0

        

    ## 画出loss随迭代次数的变化

    m = np.linspace(1,n,num = n)

    plt.plot(m,loss_list)

    plt.xlabel('iterations')

    plt.ylabel('training loss')

    plt.title('training curve')

    plt.show()

    

    ## logistic regression的结果

    plot_x1 = np.linspace(-1.0, 1.0, 100)

    plot_y1 = - (w1 * plot_x1 + w0) / w2

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x1, plot_y1, c='g')

    plt.title('result')

    plt.show()

    

    #print(w0,w1,w2)

    return y_pred1



y_pred1 = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred1 == y_total).mean())
