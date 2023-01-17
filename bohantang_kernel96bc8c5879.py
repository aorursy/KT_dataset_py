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
    bias = np.ones((x_total.shape[0], 1))
    weight = np.random.rand(x_total.shape[1],1)
    N = x_total.shape[0]
    y_total = y_total.reshape((y_total.shape[0],1))
    for i in range(n_iterations):
        y = np.dot(x_total,weight) + bias
        y_predict = 1.0 / (1.0 + np.exp(-y))
        dy = y_predict - y_total
        dw = (dy * x_total).sum(axis = 0) / N
        dw = dw.reshape((x_total.shape[1],1))
        db = dy.sum(axis = 0) / N
        db = db.reshape((1, 1))
        bias = bias - db * learning_rate
        weight = weight - dw * learning_rate
        loss = -y_total * np.log(y_predict) - (1-y_total) * np.log(1-y_predict)
        loss_list.append(np.mean(loss))
    y_predict[y_predict > 0.5] = 1
    y_predict[y_predict < 0.5] = 0
    y_predict = y_predict.reshape((y_total.shape[0],))
    return y_predict,bias,weight

y_pred, bias, weight = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())


plt.plot(np.linspace(0, n_iterations-1, n_iterations), loss_list, c='g')
plt.title('Loss')
plt.show()


plot_x = np.linspace(-1.0, 1.0, 100)
plot_x = plot_x.reshape((plot_x.shape[0],1))
plot_y = - (weight[0][0] * plot_x +bias) / weight[1][0]
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.plot(plot_x, plot_y, c='g')
plt.show()