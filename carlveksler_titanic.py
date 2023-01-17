# imports



import numpy as np # linear algebra

import pandas as pd # data processing

import seaborn as sns # visualization

import matplotlib.pyplot as plt 
# import data



path = '/kaggle/input/titanic/train_and_test2.csv'



# read data from csv



cols = list(pd.read_csv(path, nrows=1))

data = pd.read_csv(path, usecols=[i for i in cols if i[:4] != 'zero'])

n = len(data.columns) - 1



# fill null values



data.Embarked.fillna(data.Embarked.mode()[0], inplace = True)



# train-test split



train = data.sample(frac=0.8)

test = data.drop(train.index)



# standardize & normalize



train = (train - train.mean()) / train.std()

train = (train - train.min()) / (train.max() - train.min())

test = (test - test.mean()) / test.std()

test = (test - test.min()) / (test.max() - test.min())





# convert to numpy



m_train = train.shape[0]

m_test = test.shape[0]



train_y = (train['2urvived']).to_numpy().reshape(1, m_train)

train_x = (train.drop(['2urvived'], axis=1)).to_numpy().T



test_y = (test['2urvived']).to_numpy().reshape(1, m_test)

test_x = (test.drop(['2urvived'], axis=1)).to_numpy().T
# definition of the sigmoid function



def sigmoid(z):

    return 1 / (1 + np.exp(-z))
# methods for logistic regression



# forwards and backwards propagation 



def propagate_logistic(w, b, X, Y):

    

    m = X.shape[1]

    

    # current predictions made by the model

    z = np.dot(w.T, X) + b

    A = sigmoid(z) 

    # current loss of the model

    loss = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

    

    # compute dw 

    dw = (1 / m) * np.dot(X, (A - Y).T)

    # compute db

    db = (1 / m) * np.sum(A - Y)

    

    return dw, db, loss





# gradient descent



def optimize_logistic(w, b, X, Y, n_iter, alpha):

    

    loss = []

    

    for i in range(n_iter):

        

        dw, db, cur_loss = propagate_logistic(w, b, X, Y)

        w -= alpha * dw

        b -= alpha * db

        

        if i % 100 == 0:

            loss.append(cur_loss)

            if i % 1000 == 0:

                print("Loss at iteration %i: %f" % (i, cur_loss))

        

    return w, b, loss



# predict 



def predict_logistic(w, b, X):

    A = sigmoid(np.dot(w.T, X) + b)

    ret = A > 0.5

    return ret





def model_logistic(X_train, Y_train, X_test, Y_test, num_iter=2000, alpha=0.5):

    # init. weights

    w = np.random.rand(n, 1)

    b = np.random.rand()

    

    w, b, loss = optimize_logistic(w, b, X_train, Y_train, num_iter, alpha)

    

    Y_hat_test = predict_logistic(w, b, X_test)

    Y_hat_train = predict_logistic(w, b, X_train)

    

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100))

    

    return loss

    
# methods for a shallow neural network



# forwards and backwards propagation



def propagate_nn(w1, b1, w2, b2, X, Y):

    

    m = X.shape[1]

    

    # current predictions made by the model

    z1 = np.dot(w1, X) + b1

    A1 = sigmoid(z1)

    z2 = np.dot(w2, A1) + b2

    A2 = sigmoid(z2)

    

    # current loss of the model

    loss = (- 1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * (np.log(1 - A2)))

    

    # compute dw, db

    dz2 = A2 - Y

#     print('dz2 shape: ')

#     print(dz2.shape)

    dw2 = (1 / m) * np.dot(A1, dz2.T).T

#     print('dw2 shape: ')

#     print(dw2.shape)

    db2 = (1 / m) * np.sum(dz2)

#     print('db2 shape: ')

#     print(db2.shape)

    g_tag = A1 * (1 - A1)

#     print('g_tag shape: ')

#     print(g_tag.shape)

    dz1 = np.dot(w2.T, dz2) * g_tag

#     print('dz1 shape: ')

#     print(dz1.shape)

    dw1 = (1 / m) * np.dot(X, dz1.T).T

#     print('dw1 shape: ')

#     print(dw1.shape)

    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

#     print('db1 shape: ')

#     print(db1.shape)

    return dw1, db1, dw2, db2, loss





# gradient descent



def optimize_nn(w1, b1, w2, b2, X, Y, n_iter, alpha):

    

    loss = []

    

    for i in range(n_iter):

        

        dw1, db1, dw2, db2, cur_loss = propagate_nn(w1, b1, w2, b2, X, Y)

        w1 -= alpha * dw1

        b1 -= alpha * db1

        w2 -= alpha * dw2

        b2 -= alpha * db2

        

        if i % 100 == 0:

            loss.append(cur_loss)

            if i % 1000 == 0:

                print("Loss at iteration %i: %f" % (i, cur_loss))

        

    return w1, b1, w2, b2, loss



# predict 



def predict_nn(w1, b1, w2, b2, X):

    z1 = np.dot(w1, X) + b1

    A1 = sigmoid(z1)

    z2 = np.dot(w2, A1) + b2

    A2 = sigmoid(z2)

    return A2 > 0.5





def model_nn(X_train, Y_train, X_test, Y_test, num_iter=2000, alpha=0.5):

    # init. weights

    w1 = np.random.rand(8, n)

    b1 = np.random.rand(8, 1)

    w2 = np.random.rand(1, 8)

    b2 = np.random.rand()

    

    w1, b1, w2, b2, loss = optimize_nn(w1, b1, w2, b2, X_train, Y_train, num_iter, alpha)

    

    Y_hat_test = predict_nn(w1, b1, w2, b2, X_test)

    Y_hat_train = predict_nn(w1, b1, w2, b2, X_train)

    

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100))

    

    return loss
# simple logistic regression w/ gradient descent





# learning rate & iterations of gradient descent

learning_rate = 0.09

n_iter = 10000



loss = model_logistic(train_x, train_y, test_x, test_y, num_iter=10000, alpha=learning_rate)



plot = sns.lineplot(range(len(loss)), loss)

plot.set(xlabel='Iteration', ylabel='Loss')

plt.show()
# shallow neural network w/ gradient descent





# learning rate & iterations of gradient descent

learning_rate = 0.09

n_iter = 10000



loss = model_nn(train_x, train_y, test_x, test_y, num_iter=10000, alpha=learning_rate)



plot = sns.lineplot(range(len(loss)), loss)

plot.set(xlabel='Iteration', ylabel='Loss')

plt.show()