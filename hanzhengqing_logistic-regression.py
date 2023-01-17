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
def my_logistic_regression(x_total, y_total):
    # TODO
    n = np.shape(x_total)[0]
    x = []
    for i in range(n):
        x.append([x_total[i][0],x_total[i][1],1])
    x_matrix = np.mat(x)
    y_matrix = np.mat(y_total).transpose()
    weights = np.ones((3,1))
    loss_list=[]#损失函数值更新列表
    loss = np.multiply(-y_matrix, np.log(1.0 / (1 + np.exp(-x_matrix * weights))))\
           -np.multiply(np.ones((n,1))-y_matrix, np.log(np.ones((n,1))-1.0 / (1 + np.exp(-x_matrix * weights))))
    loss_list.append(np.sum(loss)/n)
    
    #更新权重
    for k in range(n_iterations): 
        y = 1.0 / (1 + np.exp(-x_matrix * weights))
        error = y_matrix - y
        weights = weights + learning_rate * x_matrix.transpose() * error
        loss = np.multiply(-y_matrix, np.log(1.0 / (1 + np.exp(-x_matrix * weights))))\
               -np.multiply(np.ones((n,1))-y_matrix, np.log(np.ones((n,1))-1.0 / (1 + np.exp(-x_matrix * weights))))
        loss_list.append(np.sum(loss)/n)    
    weights = weights.tolist()
    
    #作损失函数变化图
    plot_x = np.linspace(0, 2000, 2001)
    plot_y = loss_list
    plt.plot(plot_x, plot_y, c='g')
    plt.show()
    
    #作logistic regression结果图
    plot_x = np.linspace(-1.0, 1.0, 100)
    plot_y = - (weights[0][0] * plot_x + weights[2][0]) / weights[1][0]
    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
    plt.plot(plot_x, plot_y, c='g')
    plt.show()
    
    #输出y预测值
    y_pred = []
    y = 1.0 / (1 + np.exp(-x_matrix * weights))
    y = y.tolist()
    for i in range(n):
        if y[i][0] > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred                   
y_pred = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())