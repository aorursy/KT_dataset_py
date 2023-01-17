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

import math
from collections import Counter
    
class LogsiticRegression(object):
    weights = None
    data_num = 0
    loss_list = []
    def mySigmoid(self,x_data):
        params = - x_data.dot(self.weights)
        r = 1/(1+np.exp(params))
        return r
    def sigmod(self,Xi):
        params = - np.sum(Xi * self.weights)
        r = 1 /(1 + math.exp(params))
        return r
    def costFunction(self,Xb,y_total):
        yPre = list(self.mySigmoid(Xb))
        if(yPre.count(0) > 0 ):
            yPre.remove(0)
        if(yPre.count(1) > 0 ):
            yPre.remove(1)
        yPre = np.array(yPre)
        final_cost = y_total*np.log(yPre)+(1 - y_total)*np.log(1-yPre)
        sum0 = np.sum(final_cost)
        return -1/self.data_num * sum0
    def fit(self,x_total,y_total,alpha = 0.01,accuracy = 0.00001):
        [self.data_num,data_column] = np.shape(x_total)
        weights_num = data_column+1
        self.weights = np.full(weights_num,0.5)
        x_data = np.column_stack((np.ones((self.data_num,1)),x_total))
        count = 1
        prevJ = self.costFunction(x_data, y_total)
        self.loss_list.append(prevJ)
        while True:
            if(count >1):
                prevJ = newJ
            #注意预测函数中使用的参数是未更新的
            discend = self.mySigmoid(x_data)-y_total
            for j in range(weights_num):
                self.weights[j] = self.weights[j] -alpha * np.sum(discend * x_data[:,j])
            newJ = self.costFunction(x_data, y_total)
            self.loss_list.append(newJ)
            if newJ == prevJ or math.fabs(newJ - prevJ) < accuracy:
                print("finished")
                break
            count += 1
    def my_logsitic_regression(self,x_total,y_total):
        self.fit(x_total,y_total)
        x_data = np.column_stack((np.ones((self.data_num,1)),x_total))
        return np.round(self.mySigmoid(x_data))

            

myLogstic = LogsiticRegression()    
y_pred = myLogstic.my_logsitic_regression(x_total,y_total)
plot_x = range(1,len(myLogstic.loss_list)+1)
plt.plot(plot_x, myLogstic.loss_list, linewidth=2)
plt.xlabel("iters", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.show()
print('accuracy:',(y_pred == y_total).mean())
print('weights_x:',myLogstic.weights[1])
print('weights_y:',myLogstic.weights[2])
print('weights_b:',myLogstic.weights[0])
plot_x = np.linspace(-1.0, 1.0, 100)
plot_y = - (myLogstic.weights[1] * plot_x + myLogstic.weights[0]) /  myLogstic.weights[2]
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.plot(plot_x, plot_y, c='g')
plt.show()
