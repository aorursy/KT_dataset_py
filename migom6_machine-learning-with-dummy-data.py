import numpy as np # linear algebra
def eucleadian(pointA, pointB):

    sqDistance = 0

    print("he",pointA.shape[1])

    for i in range(pointA.shape[1]):

        print("k",pointA)

        sqDistance += np.square(pointA[i] - pointB[i])

        break

    print("no",sqDistance)

    return np.sqrt(sqDistance)
# generating data

np.random.seed(10)

Y = np.random.choice(a=range(2), size=[100,1])

X = np.random.random((100, 2))

b = np.ones([100,1]) #adding bias

X = np.concatenate((X,b), axis =1)
def perceptron(X, Y):

    weights = np.zeros([1,3])

    for j in range(X.shape[0]):

        y_predict = np.dot(weights, X[j])[0]

        y_actual = Y[j][0]

        if(not(y_predict * y_actual > 0)):

            weights = weights + Y[j]*X[j]

    return weights
perceptron(X,Y)

w.shape
w