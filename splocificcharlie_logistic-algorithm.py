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

    x_bar = np.ones((100,3))

    x_bar[:,:2]=x_total

    x = x_bar

    y = y_total

    weights = np.random.rand(3)

    def Loss(y,x,theta):

        z = np.dot(x,theta)

        p = 1/(1+np.exp(-z))

        loss = -y*(np.log10(p))-(1-y)*(np.log10(1-p))

        return np.mean(loss)

    def delta_Loss(y,x,theta):

        z = np.dot(x,theta)

        p = 1/(1+np.exp(-z))

        delta_loss =  np.dot((p-y).T,x)

        return delta_loss

    

    from tqdm import tqdm

    for i in tqdm(range(n_iterations)):

        loss = Loss(y,x,weights)

        delta_loss = delta_Loss(y,x,weights) / x.shape[0]

        weights = weights - learning_rate * delta_loss

        loss_list.append(loss)

    

    y_pred = np.dot(x,weights)

    y_pred[y_pred>=0.5] = 1

    y_pred[y_pred!=1]   = 0

    return y_pred, weights



y_pred, weights = my_logistic_regression(x_total, y_total)

print(weights[:2])

print(weights[2:3])

print('accuracy:',(y_pred == y_total).mean())



plt.plot(loss_list)

plt.title('Loss')

plt.show()

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (weights[0] * plot_x + weights[2]) / weights[1]

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plt.plot(plot_x, plot_y, c='g')

plt.show()