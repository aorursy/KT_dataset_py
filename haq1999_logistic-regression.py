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
    global n_iterations
    global learning_rate
    global loss_list
    #初始化参数
    theta1 = 0
    theta2 = 0
    theta3 = 0
    #迭代运算
    for i in range(n_iterations):
        #拟合函数y=theta1*x[0]+theta2*x[1]+theta3
        y_pre = theta1 * x_total[:,0] + theta2 * x_total[:,1] + theta3
        #计算残差
        loss = 0.5 * np.mean((y_pre-y_total)**2)
        loss_list.append(loss)
        #计算梯度
        theta1 -= learning_rate * np.mean((y_pre-y_total)*x_total[:, 0])
        theta2 -= learning_rate * np.mean((y_pre-y_total)*x_total[:, 1])
        theta3 -= learning_rate * np.mean(y_pre-y_total)
        
    y_predict = theta1 * x_total[:, 0] + theta2 * x_total[:, 1] + theta3
    y_predict[y_predict>0.5] = 1
    y_predict[y_predict<0.5] = 0
    return y_predict,theta1,theta2,theta3

y_pred,theta1,theta2,theta3 = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())

#2.
plt.plot(np.linspace(0, n_iterations-1, n_iterations), loss_list, c='G')
plt.title('Loss Curve')
plt.show()

#3.
plot_x = np.linspace(-1.0, 1.0, 100)
plot_y = - (theta1 * plot_x + theta3) / theta2 + 1
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.plot(plot_x, plot_y, c='g')
plt.show()