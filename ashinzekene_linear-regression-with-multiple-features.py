import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.datasets.samples_generator import make_regression

from matplotlib import pyplot as plt
x, y = make_regression(n_samples=200, n_features=4, noise=20, random_state=1)



print(x.shape, y.shape)


# x = (m, n) | y = (m, 1), t = (1, n), where n = features and m = features

def hyp(x, theta):

    assert x.shape[1] == theta.shape[1], "Shape x and theta different"

    return x.dot(theta.T)



def cost(x, y, theta):

    m = x.shape[0]

    x_0 = np.ones((m, 1))

    h_of_x = hyp(x, theta)

    return 1/(2 * m) * (h_of_x - y).sum() ** 2



def LinReg(x, y, theta, alpha=0.1, n_iters=400):

    m = x.shape[0]

    n = x.shape[1]

    y = y.reshape((y.shape[0], 1))

    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    assert X.shape[0] == y.shape[0], "X and y should have same no of samples"

    assert theta.shape[1] == X.shape[1], "Theta should have the same size as the number of features"

    costs = np.zeros(n_iters)

    

    

    for i in range(n_iters):

        h_of_x = hyp(X, theta)

        assert h_of_x.shape == y.shape, "H of x and y should be of the same shape"

        error = alpha * (1/m * np.sum((h_of_x - y) * X, axis=0))

        theta = theta - error

        costs[i] = cost(X, y, theta)

    return theta, costs
theta = np.zeros((1, x.shape[1] + 1))



t, costs = LinReg(x, y, theta, 0.005)



plt.plot(costs)

plt.title("GRAPH OF COST FUNCTIONS")

plt.show()

print("Theta ", t, " Costs", costs)