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

    x=[]

    epsilon = 1e-5

    b=np.ones((1,len(x_total)))

    x=np.insert(x_total,0,values=b,axis=1)

    m,n=np.shape(np.mat(x))

    thetas=np.ones((n,1))

    for i in range(n_iterations):

        h = 1 / (1.0 + np.exp(-np.mat(x)*thetas))

        error=h-np.mat(y_total).transpose()

        loss = -1 * (np.dot(np.squeeze(y_total), np.log(h)) + np.dot((1 - np.squeeze(y_total)), np.log(1 - h + epsilon)))

        loss_list.append(loss)

        thetas=thetas-learning_rate*np.mat(x).transpose()*error



    fig_loss = np.linspace(1, 2000, 2000)

    plt.plot(fig_loss, np.array(loss_list).ravel(), c='k')

    plt.xlabel('iteration')

    plt.ylabel('loss')

    plt.show()

    

    y_pred=1 / (1.0 + np.exp(-np.mat(x)*thetas))

    y_total=np.array(y_pred>0.55).astype(float).transpose()

    

    fig_x = np.linspace(-1.0, 1.0, 100)

    fig_y = -(thetas[1] * fig_x + thetas[0])/thetas[2]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(fig_x, (np.array(fig_y.tolist())).ravel(), c='g')

    plt.show()

            

    return y_total



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())