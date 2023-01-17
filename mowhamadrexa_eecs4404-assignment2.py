# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#!/usr/bin/env python

# coding: utf-8



import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

# %matplotlib inline

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

from sklearn.datasets import load_iris





def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):

#     making a list to save the number of node in each layer

    layer_units = ([len(X_train[-1])] + hidden_layer_sizes + [1])

#     defining w = 0.1 for all of the edges

    w = [np.ones((n_fan_in_ + 1, n_fan_out_)) for n_fan_in_, n_fan_out_ in

         zip(layer_units[:-1], layer_units[1:])]

    w = np.true_divide(w, 10)

#     inserting 1 at each training point's first position

    X_train = np.insert(X_train, 0, 1, axis=1)



#     for saving error of each epoch

    epoch_errors = []

    for _ in range(epochs):

        error_per_epoch = 0

#         zipping X to Y to make sure shuffling won't change the order of them

        datas = np.array(list(zip(X_train, y_train)))

#     shuffling the data

        np.random.shuffle(datas)        

        for data in datas:

#             unzip X, Y

            x, y = data

            if y == -1: y = 0

#             perform forward prop

            X, S = forwardPropagation(x, w)

#             perform back prop

            g, err = backPropagation(X, y, S, w)

#             update weights

            w = updateWeights(w, err, -alpha)

#               calculate error

            error_per_epoch += errorPerSample(X, y)

#         saving the error of the epoch

        epoch_errors.append(error_per_epoch)

    return epoch_errors, w



def forwardPropagation(x, weights):

    Xl = np.array(x)

    W = np.array(weights)

    S = []

    X = [x]

    for index, l in enumerate(W):

        wl = np.array(l)

        sl = np.transpose(wl).dot(Xl)

        Xl_before_activation = sl

#         all layers except output layer

        if index != len(W) - 1:

            activation_function = np.vectorize(activation)

            Xl = activation_function(Xl_before_activation)

            Xl = np.insert(Xl, 0, 1, axis=0)

#         last layer, output layer

        else:

            output_function = np.vectorize(outputf)

            Xl = output_function(Xl_before_activation)

        X.append(Xl)

        S.append(sl)

    return np.array(X), np.array(S)







def backPropagation(X, y_n, s, weights):

#     we need a deep copy of weights, since the elements of it are narray, without it, it will edit the source instead

    from copy import deepcopy

    w = deepcopy(weights)

#     g is in fact a vector to store all the deltas

    g = [None] * len(X)

    X = np.array(X)

    for layer, Xl in enumerate(reversed(X)):

#         calc the true value of layer number we're trying to work on

        layer = len(X) - layer - 1

#         the last layer calculation

        if layer == len(X) - 1:

            delta = 2 * (y_n - Xl[0]) * derivativeOutput(s[-1][0])

            g[layer] = np.array([delta])

#             other layer

        elif layer > 0:

#             deltas for the who layer

            deltas = []

#             calculation of the delta for the layer

            for d in range(len(s[layer - 1])):

                derivative = derivativeActivation(s[layer - 1][d])

                sum = 0

                for k, delta in enumerate(g[layer + 1]):

                    sum += (delta * w[layer][d + 1][k])

                deltas.append(sum * derivative)

            g[layer] = np.array(deltas)

#     the first element is None, we need to trim it, I added it to make the calculations easier and avoid things like: layer+1, layer-1

    g = g[1:]



#     now that we have the vector of all deltas, we need to compute the error for each edge

    to_update_W = w

    for layer, Xl in enumerate(X[:-1]):

        to_update_W[layer] = np.dot(np.array([Xl]).T, np.array([g[layer]]))

    return g, to_update_W





def updateWeights(weights, err, alpha):

#     w (new) = w (old) + a * err

    return np.subtract(np.array(weights), np.multiply(np.array(err), alpha))





def activation(s):

#     ReLu

    return 0 if s <= 0 else s





def derivativeActivation(s):

#     deriviation of the ReLu

    return 0 if s <= 0 else 1





def outputf(s):

#     sigma funtion

    return (1) / (1 + np.exp(-s))





def derivativeOutput(s):

#     derivation of the sigma

    return (outputf(s)) * (1 - outputf(s))





def errorf(x_L, y):

    if y == 1:

        return np.log(x_L)

    else:

        return -np.log(1 - x_L)





def errorPerSample(X, yn):

#     calculating the difference between the expected result and the derived result

    return errorf(X[-1][-1], yn)





def derivativeError(x_L, y):

    if y == 1:

        # derivative of np.log(x_L)

        return 1 / (x_L)

    else:

        # derivative of -np.log(1 - x_L)

        return 1 / (1 - x_L)





def pred(x_n, weights):

#     calculate forward prop and getting the very last result and return it as the answer

    x, s = forwardPropagation(x_n, weights)

    res = 1 if x[-1][-1] >= 0.5 else -1

    return res





def confMatrix(X_train, y_train, w):

    # Add implementation here



    X_train = np.insert(X_train, 0, 1, axis=1)



    y_pred = []

    for x_n in X_train:

        y_pred.append(pred(x_n, w))



    # the confusion maxtrix that we will return

    # matrix = [[0, 0], [0, 0]]

    matrix = np.zeros((2, 2), np.int8)



    # Populating our matrix using the prediction data

    for index, y in enumerate(y_train):

        if y == -1 and y_pred[index] == -1:

            matrix[0][0] += 1

        elif y == -1 and y_pred[index] == 1:

            matrix[0][1] += 1

        elif y == 1 and y_pred[index] == -1:

            matrix[1][0] += 1

        else:

            matrix[1][1] += 1



    # returning the result

    return matrix

    # return confusion_matrix(y_train, y_pred)





def plotErr(e, epochs):

    x_axis = range(1, epochs + 1)

    plt.plot(e)

    plt.show()





def test_SciKit(X_train, X_test, Y_train, Y_test):

    nn = MLPClassifier(hidden_layer_sizes=(300, 100), random_state=1, alpha=10 ** (-5))

    nn.fit(X_train, Y_train)

    pred = nn.predict(X_test)

    cm = confusion_matrix(Y_test, pred)

    return cm





def test():

    X_train, y_train = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size=0.2)



    for i in range(80):

        if y_train[i] == 1:

            y_train[i] = -1

        else:

            y_train[i] = 1

    for j in range(20):

        if y_test[j] == 1:

            y_test[j] = -1

        else:

            y_test[j] = 1





    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)

    plotErr(err, 100)

    cM = confMatrix(X_test, y_test, w)



    sciKit = test_SciKit(X_train, X_test, y_train, y_test)



    print("Confusion Matrix is from Part 1a is:\n", cM)

    print("Confusion Matrix from Part 1b is:\n", sciKit)





test()
