import numpy as np

import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float') 

y_total = lines[:, 3].astype('float') #标签



pos_index = np.where(y_total == 1) # pos_index 标记1的下标

neg_index = np.where(y_total == 0) # neg_index 标记0的下标

plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r') #red 1

plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b') #blue 0

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

    #初始化参数

    [N,n] = x_total.shape #矩阵大小

    w = np.zeros(n)

    b = 0

    loss_list=np.zeros(n_iterations) #存储loss

    

    for epoch in range(n_iterations):

        logits = w.dot(x_total.T) + b

        fx =  1/(1 + np.exp(-logits))

        loss_list[epoch] = -np.sum(y_total*np.log(fx)+(1-y_total)*np.log(1-fx))/N

        partialD_w =  np.dot((fx - y_total), x_total) / N

        partialD_b =  np.sum(fx - y_total) / N        

        #更新参数

        w = w - learning_rate * partialD_w

        b = b - learning_rate * partialD_b

    

    #loss曲线图    

    plt.plot(list(range(n_iterations)), loss_list)    

    plt.xlabel('iterations')

    plt.ylabel('training loss')

    plt.title('training curve figure')

    plt.show()

    

    #分类结果图

    my_plot_x = np.linspace(-1.0, 1.0, 100)

    my_plot_y = - (w[0] * my_plot_x + b) / w[1]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(my_plot_x, my_plot_y, c='g')

    plt.title('the result of logistic regression')

    plt.show()

    

    #打印参数

    print(w)

    print([b])

    return np.round(fx)  #返回四舍五入的预测值



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())