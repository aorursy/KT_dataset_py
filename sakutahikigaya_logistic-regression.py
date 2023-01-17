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

    return 1 / (1 + np.exp(-z))

def my_logistic_regression(x_total, y_total):

    # TODO

    weights = np.ones((3, 1))  #初始化回归系数（3, 1)

    dataIn = np.insert(x_total, 0, 1, axis=1)  #特征数据集，构造常数项x0

    data_mat = np.mat(dataIn)

    label_mat = np.mat(y_total).transpose()

    for i in range(n_iterations):

        h = sigmoid(data_mat * weights)  # sigmoid 函数

        weights = weights + learning_rate * data_mat.transpose() * (label_mat - h)  # 梯度

        loss_list.append (float(- label_mat.T * (np.log10(h) )- (- label_mat + 1).T * (np.log10(1 - h))))

    plot_loss = [i for i in range(0, n_iterations)]

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y = - (weights[1, 0] * plot_x + weights[0, 0]) / weights[2, 0]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()



    plt.plot(plot_loss, loss_list, c='b')

    plt.show()       

    

    y_pred = (np.array(sigmoid(data_mat * weights))).reshape(100,) 

    yt_index = np.where(y_pred >= 0.5)

    yf_index = np.where(y_pred < 0.5)

    

    y_pred[yt_index] = 1

    y_pred[yf_index] = 0

    return y_pred



y_pred = my_logistic_regression(x_total, y_total)



print('accuracy:',(y_pred == y_total).mean())
