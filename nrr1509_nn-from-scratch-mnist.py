from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from scipy.io import loadmat

mnist = loadmat("../input/mnist-original.mat")

mnist_data = mnist["data"].T

mnist_label = mnist["label"]
X=mnist_data

y=mnist_label

y = y.reshape(70000, 1)

X = X / 255

import numpy as np



y_new = np.zeros(y.shape)

y_new[np.where(y == 0.0)[0]] = 1

y = y_new

m = 60000

m_test = X.shape[0] - m



X_train, X_test = X[:m].T, X[m:].T

y_train, y_test = y[:m].reshape(1,m), y[m:].reshape(1,m_test)

np.random.seed(138)

shuffle_index = np.random.permutation(m)

X_train, y_train = X_train[:,shuffle_index], y_train[:,shuffle_index]
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



i = 3

plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)

plt.axis("off")

plt.show()

print(y_train[:,i])
def sigmoid(z):

    s = 1 / (1 + np.exp(-z))

    return s
def compute_loss(Y, Y_hat):



    m = Y.shape[1]

    L = -(1./m) * ( np.sum( np.multiply(np.log(Y_hat),Y) ) + np.sum( np.multiply(np.log(1-Y_hat),(1-Y)) ) )



    return L
learning_rate = 1



X = X_train

Y = y_train



n_x = X.shape[0]

m = X.shape[1]



W = np.random.randn(n_x, 1) * 0.01

b = np.zeros((1, 1))



for i in range(2000):

    Z = np.matmul(W.T, X) + b

    A = sigmoid(Z)



    cost = compute_loss(Y, A)



    dW = (1/m) * np.matmul(X, (A-Y).T)

    db = (1/m) * np.sum(A-Y, axis=1, keepdims=True)



    W = W - learning_rate * dW

    b = b - learning_rate * db



    if (i % 100 == 0):

        print("Epoch", i, "cost: ", cost)



print("Final cost:", cost)
A.shape
X = X_train

Y = y_train



n_x = X.shape[0]

n_h = 64

learning_rate = 1#Chosen arbitrarily, it decides by what scale the weigths and bias should  change



W1 = np.random.randn(n_h, n_x)#Weights of nodes connecting i/p layer and first hidden layer

b1 = np.zeros((n_h, 1))

W2 = np.random.randn(1, n_h)#Weights of nodes connecting second layer and o/p layer

b2 = np.zeros((1, 1))



for i in range(2000):



    Z1 = np.matmul(W1, X) + b1

    A1 = sigmoid(Z1)#o/p of first hidden layer

    Z2 = np.matmul(W2, A1) + b2

    A2 = sigmoid(Z2)##o/p of second hidden layer



    cost = compute_loss(Y, A2)



    dZ2 = A2-Y

    dW2 = (1./m) * np.matmul(dZ2, A1.T)

    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)



    dA1 = np.matmul(W2.T, dZ2)

    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))

    dW1 = (1./m) * np.matmul(dZ1, X.T)

    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)



    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1



    if i % 100 == 0:

        print("Epoch", i, "cost: ", cost)



print("Final cost:", cost)
X=mnist_data

y=mnist_label

y = y.reshape(70000, 1)

X = X / 255
digits = 10

examples = y.shape[0]

y = y.reshape(1, examples)

Y_new = np.eye(digits)[y.astype('int32')]

Y_new = Y_new.T.reshape(digits, examples)



m = 60000

m_test = X.shape[0] - m

X_train, X_test = X[:m].T, X[m:].T

Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]

shuffle_index = np.random.permutation(m)

X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]



i = 12

plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)

plt.axis("off")

plt.show()

Y_train[:,i]
def compute_multiclass_loss(Y, Y_hat):



    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))

    m = Y.shape[1]

    L = -(1/m) * L_sum



    return L
n_x = X_train.shape[0]

n_h = 64

learning_rate = 1



W1 = np.random.randn(n_h, n_x)

b1 = np.zeros((n_h, 1))

W2 = np.random.randn(digits, n_h)

b2 = np.zeros((digits, 1))



X = X_train

Y = Y_train



for i in range(2000):



    Z1 = np.matmul(W1,X) + b1

    A1 = sigmoid(Z1)

    Z2 = np.matmul(W2,A1) + b2

    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)



    cost = compute_multiclass_loss(Y, A2)



    dZ2 = A2-Y

    dW2 = (1./m) * np.matmul(dZ2, A1.T)

    db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)



    dA1 = np.matmul(W2.T, dZ2)

    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))

    dW1 = (1./m) * np.matmul(dZ1, X.T)

    db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)



    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1



    if (i % 100 == 0):

        print("Epoch", i, "cost: ", cost)



print("Final cost:", cost)