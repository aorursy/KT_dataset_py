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

deta=np.zeros(3)

n_iterations = 2000

learning_rate = 0.1

loss_list = []



def my_logistic_regression(x_total, y_total):

    mid=np.ones((x_total.shape[0],1))

    x_total=np.concatenate((mid,x_total),axis=1)

    global deta

    for i in range(2000):

        y=np.dot(x_total,deta)

        y_tra=1.0/(1.0+np.exp(-y))

        #梯度计算

        gra=np.dot(x_total.T,(y_tra-y_total))

        #更新deta

        deta=deta-gra*learning_rate

        #计算损失

        loss=-y_total*np.log(y_tra)-(1-y_total)*np.log(1-y_tra)

        loss_list.append(np.mean(loss))

        y_tra[y_tra>0.5]=1

        y_tra[y_tra<0.5]=0

        

   

    return y_total



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())

plot_x = np.linspace(0,n_iterations-1,2000)

plt.plot(plot_x, loss_list, c='g')

plt.xlabel("iterations")

plt.ylabel("loss")

plt.show()



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (deta[1] * plot_x + deta[0]) /  deta[2]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()