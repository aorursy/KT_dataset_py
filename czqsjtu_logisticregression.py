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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cal_loss(x,y,w):
    m,n = np.shape(x)
    loss = 0.0
    for i in range(m):
        sum_theta = 0
        for j in range(n):
            sum_theta +=x[i,j]*w.T[0,j]
        prob = sigmoid(sum_theta)

        loss += - y[i,0] * np.log(prob)- (1-y[i,0])*np.log(1-prob)
    return loss

def my_logistic_regression(x_total, y_total):
    # TODO
    data = np.mat(x_total)
    label = np.mat(y_total).T
    m,n = np.shape(data)
    weight = np.ones((n,1))
    eps = 0.0001
    count = 0
    
    for i in range(n_iterations):
       
        loss = cal_loss(data,label,weight)
        
        h = sigmoid(data*weight)
        e = h-label
        newWeight = weight - learning_rate * data.T * e
       
        newLoss = cal_loss(data,label,newWeight)
        loss_list.append(newLoss)
        
        if abs(newLoss -loss)<eps :
            break
        else:
            weight = newWeight
            count +=1
    
    
    plt.plot(range(count+1),loss_list)
    print(weight)
    
    plot_x = np.linspace(-1.0, 1.0, 100)
    plot_y = - (weight[0] * plot_x ) /  weight[1]
    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
    plt.plot(plot_x, plot_y, c='g')
    plt.show()
    
    return y_total

y_pred = my_logistic_regression(x_total, y_total)

print(weight)
plot_x = np.linspace(-1.0, 1.0, 100)
plot_y = - (myLogstic.weights[1] * plot_x + myLogstic.weights[0]) /  myLogstic.weights[2]
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')
plt.plot(plot_x, plot_y, c='g')
plt.show()