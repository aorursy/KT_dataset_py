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

def sigmod(input):
    return 1.0/(1+np.exp(-input))

def my_logistic_regression(x_total, y_total):
    x = np.mat(x_total)
    y = np.mat(y_total)
    yt = y.transpose()
    x_num,x_pos = np.shape(x)
    w = np.ones((2,1))
    b = 0
    
    for i in range(n_iterations):
        h = sigmod(x*w+b)
        loss = -np.multiply(yt,np.log10(h)) - np.multiply((1-yt),np.log10(1-h))
        loss_list.append(np.sum(loss)/x_num)
        dis = yt - h
        w = w + learning_rate*x.transpose()*dis/x_num
        b = b + learning_rate*dis/x_num
        yt = sigmod(x*w+b)
        
        yt[yt>0.5]=1
        yt[yt<=0.5]=0
        y = yt.transpose()
        
    plot_x = np.linspace(-1.0, 1.0, 100)
    plot_y = -(b + w[0]*plot_x)/w[1]
    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
    plt.plot(plot_x, plot_y, c='g')
    plt.show()
    
    return y

y_pred = my_logistic_regression(x_total, y_total)
print('accuracy:',(y_pred == y_total).mean())


plot_x2 = np.linspace(0,n_iterations-1,n_iterations)
plt.plot(plot_x2,loss_list,c='g')
plt.show()