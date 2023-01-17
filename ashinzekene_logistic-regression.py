import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from scipy.special import expit
# x: (m x n), theta: (n, 1), y: (m, 1)



def sigmoid(X, theta):

    theta_x = X.dot(theta)

    return expit(theta_x)





def cost_function(y, theta, h):

    assert h.shape == y.shape, "h and y must be of the same shape, h (%s %s)" % h.shape

    m = y.shape[0]



    y_0 = -y * np.log(h)

    y_1 = - (1 - y) * np.log(1 - h)

    return 1/m * np.sum(y_0 + y_1)



def logisitcRegression(X, y, theta, alpha=0.05, n_iters=500):

    assert X.shape[0] == y.shape[0]

    assert X.shape[1] == theta.shape[0] 



    m = y.shape[0]

    costs = np.zeros(n_iters)

    

    for i in range(n_iters):

        h = sigmoid(X, theta)

        gradient = X.T.dot(h - y)/ m

        

        assert gradient.shape == theta.shape, "Theta and gradient must be of the same shape. G (%s %s)" % gradient

        theta = theta - gradient

        

        costs[i] = cost_function(y, theta, h)

    

    return theta, costs



def predict(theta, x_test):

    m = x_test.shape[0]

    result = x_test.dot(theta)

    outcome = np.zeros(m)

    for i in range(m):

        if result[i] > 0.5:

            outcome[i] =1

        else:

            outcome[i] = 0

    return outcome

        

    
data = pd.read_csv("../input/ex2data1.txt", names=["x1","x2","y"])

data.head()
y_data = data.filter("y")

x_data = data.drop("y", axis=1)

print(y_data.head())

print(x_data.head())
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x = sc.fit_transform(x_data)

x[:2]

y = y_data.values
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

x_train.shape, y_test.shape
intercept = np.ones((x_train.shape[0], 1))

X_train = np.concatenate((intercept, x_train), axis=1)

n = X.shape[1]



theta = np.zeros((n, 1))



t, costs = logisitcRegression(X_train, y_train, theta)

t
plt.plot(costs)
y_train_pred = predict(t, X_train)

plt.figure(figsize=(20,4))

plt.plot(y_train, "bo", ms=9)

plt.plot(y_train_pred, "ro")
X_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)



y_test_pred = predict(t, X_test)



l = list(range(y_test.shape[0]))

plt.plot(l, y_test, "bo", ms="8")

plt.plot(l, y_test_pred, "ro")
from sklearn.linear_model import LogisticRegression





reg = LogisticRegression(solver="liblinear")



reg.fit(x_train, y_train)



print("Coeff", reg.coef_)

print("Intercept", reg.intercept_)
y_test_pred = reg.predict(x_test)



l = list(range(y_test.shape[0]))

plt.plot(l, y_test, "bo", ms="8")

plt.plot(l, y_test_pred, "ro")
y_train_pred = reg.predict(x_train)



plt.figure(figsize=(20, 4))



plt.plot(y_train, "bo", ms="8")

plt.plot(y_train_pred, "ro")