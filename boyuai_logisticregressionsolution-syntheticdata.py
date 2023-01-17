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

print('数据集大小:', x_total.shape[0])
n_iterations = 1000

learning_rate = 0.1



def sigmoid(z):

    result = 1.0 / (1.0 + np.exp(-1.0 * z))

    return result



weight = np.zeros(3)

x_total_concat = np.hstack([x_total, np.ones([x_total.shape[0], 1])])



for i in range(n_iterations):

    loss = (- y_total * np.log(sigmoid(np.dot(x_total_concat, weight))) - (1 - y_total) * (np.log(1 - sigmoid(np.dot(x_total_concat, weight))))).mean()

    if i % 20 == 0:

        print('current num_step:', i)

        print('loss:', loss) 

    

    w_gradient = (x_total_concat * np.tile((sigmoid(np.dot(x_total_concat, weight)) - y_total).reshape([-1, 1]), 3)).mean(axis=0)

    weight = weight - learning_rate * w_gradient 

    

y_pred = np.where(np.dot(x_total_concat, weight)>0, 1, 0)

print('accuracy:',(y_pred == y_total).mean())
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

plot_x = np.linspace(-1.0, 1.0, 100)

plot_y = - (weight[0] * plot_x + weight[2]) / weight[1]

plt.plot(plot_x, plot_y, c='g')

plt.show()