# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



print("ready")
data = pd.read_csv('../input/ex1data1.txt', sep= ',', header=None)

data.head()

data.columns
X = pd.DataFrame(data[0])

print(X)

y = data[1]

m = y.shape[0]

plt.scatter(X, y)

plt.xlabel("Population in 10 Ks")

plt.ylabel("Profit in 10 Ks")
#add a column of ones in X in order to simulate the first feature / interceptor and create a dataframe for X

def initialize_dataset(X):

    X_test = pd.DataFrame()

    X_test[0] = np.ones(m)

    for col in X.columns:

        X_test[col + 1] = X[col]

    return X_test

X_test = initialize_dataset(X)

X_test.head()
#initaliaze parameters for linear regression, iterations and learning rate

def initialize_parameters(X, iterations, learning_rate):

    theta = pd.Series(np.zeros(X.shape[1]))

    iterations = iterations

    alpha = learning_rate

    return theta, iterations, alpha

theta, iterations, alpha = initialize_parameters(X_test, 1500, 0.01)

def compute_predictions(X, theta):

    return X.dot(theta)
#create the cost function

def compute_cost(X, y, theta):

    predictions = compute_predictions(X, theta)

    return (predictions - y).pow(2).sum()/ (2 * m)

cost = compute_cost(X_test, y, theta)

print(cost)
#gradient descent implementation

theta, iterations, alpha = initialize_parameters(X_test, 1500, 0.01)

def fit_model(X, y, theta, alpha, iterations):

    list_of_cost = []

    for i in range(iterations):

        predictions = compute_predictions(X, theta)

        #update theta parameters on each iteration

        theta = theta - (alpha / m) * X.T.dot(predictions - y)

        #save the cost on each iterations to plot later

        list_of_cost.append(compute_cost(X, y, theta))

    cost_history = pd.DataFrame({"number_of_iterations": range(iterations), "cost": list_of_cost})

    return theta, cost_history

theta, cost_history = fit_model(X_test, y, theta, alpha, iterations)
#see the cost evolution 

cost_history["cost"].plot.line()

plt.xlabel("Number of Iterations")

plt.ylabel("Cost")
#plot the regression line

predictions = compute_predictions(X_test, theta)

plt.scatter(X, y)

plt.xlabel('Population of City in 10K')

plt.ylabel('Profit in $10Ks')

plt.plot(X, predictions, color="red")

data_2 = pd.read_csv('../input/ex1data2.txt', sep= ',', header=None)

data_2.head()
#Set up X and y datasets

last_column = data_2.columns[-1]

X = data_2.drop(last_column, axis=1)

y = data_2[last_column]

m = y.shape[0]
#Normalize feature

def normalize_features(X):

    means = X.mean()

    X = X - X.mean()

    deviation = X.std()

    X = X / X.std()

    return X, means, deviation

X, means, deviation = normalize_features(X)

print(X.head(), means, deviation)
#fit the model

X_test = initialize_dataset(X)

theta, iterations, alpha = initialize_parameters(X_test, 50, 0.5)

X_test.head()

theta, cost_history = fit_model(X_test, y, theta, alpha, iterations)

cost_history["cost"].plot.line()

plt.xlabel("Number of Iterations")

plt.ylabel("Cost")