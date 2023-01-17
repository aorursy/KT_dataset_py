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

learning_rate = 0.01

loss_list = []

epochs_list = []



def my_logistic_regression(x_total, y_total):

    x_total = np.insert(x_total,0,1.0,1)

    xMat = np.mat(x_total)

    yMat = np.mat(y_total).transpose()

    m,n = xMat.shape

    weights = np.ones((n,1)) #初始化模型参数

    epochs_count = 0

    while epochs_count<n_iterations:

        h = 1.0/(1+np.exp(-xMat*weights)) #预测值

        error = h - yMat  #预测值与实际值差值

        grad = (xMat.transpose()*error) #损失函数的梯度

        weights = weights - learning_rate*grad #参数更新

        loss = -1*(np.dot(y_total,np.log(h))+np.dot((1-(y_total)),np.log(1-h)))   #当前损失

        loss_list.append(loss)

        epochs_list.append(epochs_count)

        epochs_count += 1        



    y_pred = 1.0/(1+np.exp(-xMat*weights))

    y_total = np.array(y_pred>0.55).astype(float).transpose()

    return y_total,weights



y_pred,weights = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())

loss_list = np.squeeze(loss_list)

plt.figure()

plt.plot(epochs_list,loss_list)   #损失曲线

plt.xlabel('epochs')

plt.ylabel('loss')



plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = (-weights[0,0]-weights[1,0]*plot_x)/weights[2,0]

plt.figure()

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x,plot_y,c='g')



plt.show()