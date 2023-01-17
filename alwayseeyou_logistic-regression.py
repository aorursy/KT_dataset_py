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
import math
def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))
def my_logistic_regression(x_total, y_total):
    x_total = np.mat(x_total)
    y_total = np.mat(y_total).T
    m,n = np.shape(x_total)    
    weights = np.ones((n,1))  #3*1     
    for i in range(n_iterations):
        h = 1.0/(1+np.exp(np.dot(x_total,weights))) # 100*3 * 3*1 =  100*1
        L = (h - y_total) #100*1
        weights = weights - learning_rate * x_total.T * L # 3*1 - 3*100 * 100*1
   
    fig = plt.figure()
    ax = fig.add_subplot(111)#1行1列 
    ax.scatter(x_total[pos_index, 0].tolist(), x_total[pos_index, 1].tolist(), marker='o', c='r')
    ax.scatter(x_total[neg_index, 0].tolist(), x_total[neg_index, 1].tolist(), marker='x', c='b')
 
    x = np.linspace(-1.0, 1.0, 100)
    y = (-1.0-weights[0]*x)/(weights[1])
    y = y.T
    ax.plot(x, y, 'k--',color = 'black', linewidth=2)
    plt.xlabel('Logistics Regression GradDescent')
    plt.show()

    count = 0
    for i in range(m):
        h = 1.0/(1+np.exp(x_total[i,:] * weights))
        if ( h<0.5 and int(y_total[i,0]) == 1) or ( h>0.5 and int(y_total[i,0]) == 0 ):
            count += 1 
    return count/m

y_pred = my_logistic_regression(x_total, y_total)
print('accuracy:',y_pred)