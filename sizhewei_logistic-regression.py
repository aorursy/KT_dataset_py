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
# without bias



# 1. finish function my_logistic_regression;

# 2. draw a training curve (the x-axis represents the number of training iterations, and the y-axis represents the training loss for each round);

# 3. draw a pic to show the result of logistic regression (just like the pic in section 2);



n_iterations = 2000

learning_rate = 0.1

loss_list = []



def mysigma(x, weights): # return (1,num_features)

    return 1/(1+np.exp(-np.dot(weights,(x.T))))



def mylossentropy(x_total, y_total, weights):

    num_sample = x_total.shape[0]

    num_feature = x_total.shape[1]

    sigma = mysigma(x_total, weights)

    y_total_proc = np.expand_dims(y_total, axis=1).T # y_total_proc.shape = (1, num_features)

    

    loss = -np.dot(y_total_proc,np.log(sigma).T) - np.dot((1-y_total_proc),np.log(1-sigma).T)

    

    weights = weights - learning_rate*np.dot((sigma - y_total_proc),x_total)/num_sample

    

    return loss[0][0], weights



def my_logistic_regression(x_total, y_total):

    # TODO

    num_sample = x_total.shape[0]

    num_feature = x_total.shape[1]

    # print(num_sample)

    # print(num_feature)

    weights = np.zeros((1,num_feature)) + 1/num_feature  # weights.shape = (1,num_feature)

#     bias = np.zeros((1,1))

    for i in range(n_iterations):

        loss, weights = mylossentropy(x_total, y_total, weights)

        loss_list.append(loss)

        if i%50 == 0:

            print('{0} loss is: {1}'.format(i,loss))

        if i%500 == 0:

            print('{0} weights is: {1}'.format(i, weights))

        

    y_pred = (mysigma(x_total, weights)>0.5)

    

    return y_pred,weights



x_total_proc = np.c_[np.ones((x_total.shape[0],1)),x_total]

# print(x_total_proc.shape)

y_pred, weights = my_logistic_regression(x_total_proc, y_total)

print('accuracy:',(y_pred == y_total).mean())
x = np.arange(0,2000)

plt.plot(x,loss_list,label = 'linear')



plt.xlabel('time')       # 梯度下降的次数

plt.ylabel('loss')       # 损失值

plt.title('loss trend')         # 损失值随着W不断更新，不断变化的趋势

plt.legend()               # 图形图例

plt.show()
plot_x = np.linspace(-1.0, 1.0, 2000)

plot_y = (-weights[0][0]-weights[0][1]*plot_x)/weights[0][1]

print(plot_x.shape)

print(plot_y.shape)

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()
# # 1. finish function my_logistic_regression;

# # 2. draw a training curve (the x-axis represents the number of training iterations, and the y-axis represents the training loss for each round);

# # 3. draw a pic to show the result of logistic regression (just like the pic in section 2);



# n_iterations = 2000

# learning_rate = 0.1

# loss_list = []



# def mysigma(x, weights, bias): # return (1,num_features)

#     return 1/(1+np.exp(-np.dot(weights,(x.T))+bias))



# def mylossentropy(x_total, y_total, weights, bias):

#     num_sample = x_total.shape[0]

#     num_feature = x_total.shape[1]

#     sigma = mysigma(x_total, weights, bias)

#     y_total_proc = np.expand_dims(y_total, axis=1).T # y_total_proc.shape = (1, num_features)

    

#     loss = -np.dot(y_total_proc,np.log(sigma).T) - np.dot((1-y_total_proc),np.log(1-sigma).T)

    

#     weights = weights - learning_rate*np.dot((sigma - y_total_proc),x_total)/num_sample

#     bias = bias - learning_rate*np.mean(sigma - y_total_proc)/num_sample

    

#     return loss[0][0], weights, bias



# def my_logistic_regression(x_total, y_total):

#     # TODO

#     num_sample = x_total.shape[0]

#     num_feature = x_total.shape[1]

#     # print(num_sample)

#     # print(num_feature)

#     weights = np.zeros((1,num_feature)) + 1/num_feature  # weights.shape = (1,num_feature)

#     bias = np.zeros((1,1))

#     for i in range(n_iterations):

#         loss, weights, bias = mylossentropy(x_total, y_total, weights, bias)

#         loss_list.append(loss)

#         if i%50 == 0:

#             print('{0} loss is: {1}'.format(i,loss))

#         if i%500 == 0:

#             print('{0} weights is: {1}'.format(i, weights))

        

#     y_pred = (mysigma(x_total, weights, bias)>0.5)

    

#     return y_pred,weights,bias



# y_pred, weights, bias = my_logistic_regression(x_total, y_total)

# print('accuracy:',(y_pred == y_total).mean())