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

import pandas as pd

import numpy.random



pdData = pd.read_csv('../input/data.csv',header=None,names=['A','B','E'])

pdData.insert(0,'Ones',1)

orig_data = pdData.as_matrix() 

cols = orig_data.shape[1]

X = orig_data[:,0:cols-1]

y = orig_data[:,cols-1:cols]

theta = np.zeros([cols-1,1])



n_iterations = 2000

learning_rate = 0.1

loss_list = []

def sigmoid(a):

    return 1 / (1 + np.exp(-a))



def model(X,theta):

    return sigmoid(np.matmul(X,theta))



def costFunction(X,y,theta):

    left = np.multiply(-y,np.log(model(X,theta))) 

    right = np.multiply((1-y),np.log(1-model(X,theta)))

    return np.sum(left-right)/(len(X))



def gradient(X,y,theta):

    grad = np.zeros(theta.shape)

    error = np.matmul(X.T,(model(X,theta)-y))

    grad = error/len(X)

    return grad



def shuffleData(data):

    np.random.shuffle(data)

    X = data[:, 0:cols-1]

    y = data[:, cols-1:]

    return X, y



def descent(data, theta, n_iterations, learning_rate):

    i = 0 #  迭代次数

    k = 0 # batch

    X,y = shuffleData(data)

    grad = np.zeros(theta.shape) # 计算梯度

    loss_list = [costFunction(X,y,theta)] # 损失值

    

    while True:

        grad = gradient(X[k:k+100],y[k:k+100],theta)

        X,y = shuffleData(data) # 重新洗牌

        theta = theta - learning_rate*grad # 参数更新

        loss_list.append(costFunction(X,y,theta)) # 计算新的损失

        i += 1

        if i > n_iterations: break

        

    return theta,loss_list,grad



def my_logistic_regression(data, theta, n_iterations, learning_rate):

    # TODO

    theta, loss_list, grad = descent(data, theta, n_iterations, learning_rate)

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(np.arange(len(loss_list)), loss_list, 'r')

    ax.set_xlabel('Iterations')

    ax.set_ylabel('Loss')

    return theta



def predict(X, theta):

    return [1 if x >= 0.5 else 0 for x in model(X, theta)]



X = orig_data[:, :3]

y = orig_data[:, 3]

theta_final = my_logistic_regression(orig_data, theta, n_iterations, learning_rate)

predictions = predict(X, theta_final)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]

accuracy = (sum(map(int, correct)) % len(correct))

print ('accuracy = {0}%'.format(accuracy))



coef1 = - theta_final[0,0] / theta_final[2,0]

coef2 = - theta_final[1,0] / theta_final[2,0]



x = np.linspace(20,100,100)

y_predict = coef1 + coef2 * x

fig,ax = plt.subplots()

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()