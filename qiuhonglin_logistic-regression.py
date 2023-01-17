import numpy as np



import matplotlib.pyplot as plt



lines = np.loadtxt('../input/data.csv', delimiter=',', dtype='str')

x_total = lines[:, 1:3].astype('float')

y_total = lines[:, 3].astype('float')

print(x_total)

#print(y_total)

print(x_total[1,1])

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

print(lr_clf.coef_[0][1])

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

x_total1=[]



def sigmoid(inx):

    

    return 1.0/(1+np.exp(-inx))





def my_logistic_regression(x_total, y_total):

    

    import matplotlib.pyplot as plt

   

    

   

    x_totalmat=np.mat(x_total)

    y_totalmat=np.mat(y_total).transpose()

    m,n=np.shape(x_totalmat)

  

    b=0;

    weights=np.ones((n,1))

    

   

    for k in range (n_iterations):

        t=x_totalmat*weights+b

        h=sigmoid(t)

        loss=-np.multiply(y_totalmat,np.log(h))-np.multiply((1-y_totalmat),np.log(1-h))

        loss_list.append(np.sum(loss)/m)

        weights+=learning_rate*(x_totalmat.transpose())*(y_totalmat-h)/m

        b+=learning_rate*np.sum(y_totalmat-h)/m

    y_totalmat=x_totalmat*weights+b

    y_totalmat=sigmoid(y_totalmat)

    y_totalmat[y_totalmat>=0.5]=1

    y_totalmat[y_totalmat<0.5]=0

    plot_x = np.linspace(-1.0, 1.0, 100)

    plot_y =(-b-weights[0]*plot_x)/weights[1]

    plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker='o', c='r')

    plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker='x', c='b')

    plt.plot(plot_x, plot_y, c='g')

    plt.show()

    plt.plot(np.linspace(0, n_iterations-1, n_iterations), loss_list, c='b')

    plt.show()

    # TODO

    y_totalmat=y_totalmat.transpose()

    return y_totalmat



y_pred = my_logistic_regression(x_total, y_total)

print('accuracy:',(y_pred == y_total).mean())