import numpy as np

import matplotlib.pyplot as plt
def loadData(fileName):

    print(fileName)

    data = np.matrix(np.loadtxt(fileName, delimiter=',', skiprows = 1))

    return (data[:, 0], data[:, 1])
def GradientDissent(X, Y, theta, alpha, num_iters):

    m = np.size(Y)

    hx = np.matmul(X, theta)

    c1 = Cost(theta, X, Y)

    for i in range(0, num_iters):

        temp = (alpha / m) * np.matmul(X.T, (hx - Y))

        c2 = Cost(theta - temp, X, Y)

 

        if c1 > c2:

            c1 = c2

            theta = theta - temp

        #print (c1, c2, theta)

    

    return theta
def Cost(theta, X, Y):

    m = Y.size

    hx = np.matmul(X, theta)

    return np.sum(np.power(np.subtract(hx, Y), 2)) / (2 * m)
X, Y = loadData("../input/train/train.csv")

m = Y.size

t = X
X = np.hstack((np.matrix(np.ones(m).reshape(m, 1)), t))



theta = np.matrix(np.ones(2).reshape(2, 1))

theta = GradientDissent(X, Y, theta, 0.000001, 100)

Cost(theta, X, Y)
print(theta)

plt.scatter(np.array(t), np.array(Y), marker = 'x', color='r')

plt.plot(np.array(t),np.array(np.matmul(X, theta)))

plt.title("After Regression: slope = " + str(theta))

plt.show()

print(Cost(theta, X, Y))
#test data

X, Y = loadData('../input/train/test.csv')

m = Y.size

t = X

X = np.hstack((np.matrix(np.ones(m).reshape(m, 1)), t))

plt.scatter(np.array(t), np.array(Y), marker = 'x', color='r')

plt.plot(np.array(t),np.array(np.matmul(X, theta)))

plt.title("After Regression: slope = " + str(theta))

plt.show()

print(Cost(theta, X, Y))