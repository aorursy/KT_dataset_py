# Import Statements

import os



import numpy as np # linear algebra

import pandas as pd # data processing



import matplotlib.pyplot as plt

%matplotlib inline
# Global Variables

dataset_path = "/kaggle/input/house-prices/house-prices.txt"

column_names = ['house_size','num_of_bedrooms','price']
# Loading Data

data = pd.read_csv(dataset_path, header=None)

data.columns =(column_names)

data.head()
def plot2DScatter(X, Y, title=None, x_label=None, y_label=None):

    '''

    Simplified version of the scatter plot present in matplotlib library. It creates a scatter plot from the given points.

    

    Parameters

    ----------

    X : iterable (like python list, numpy array, pandas series, etc.)

        x-coordinate of the points to be plotted

    Y : iterable

        y-coordinate of the points to be plotted

    title : string, optional

        title of the plot

    x_label : string, optional

        label of x-axis

    y_label : string, optional

        label of y-axis

    '''

    plt.plot(X, Y, '.')

    plt.title(title)

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.show()
plot2DScatter(data.house_size, data.price, "House Size Vs. Price", "House Size", "Price")
plot2DScatter(data.num_of_bedrooms, data.price, "Number of Bedrooms Vs. Price", "Number of Bedrooms", "Price")
def hypothesis_func(parameter_arg, x):

    return np.dot(np.concatenate((np.array([1]).reshape(1,1), x)).T, parameter_arg)



def cost_func(parameter_arg, X, Y):

    cost = 0

    for j in range(X.shape[0]):

        cost += (hypothesis_func(parameter_arg, X[j,:].reshape(X.shape[1],1)) - Y[j]) ** 2

    return cost / 2



def GDStep(parameter_arg, lr, X, Y):

    new_parameters = []

    dim = X.shape[1]

    for i in range(parameter_arg.shape[0]):

        grad = 0

        for j in range(X.shape[0]):

            if i == 0:

                grad += (hypothesis_func(parameter_arg, X[j,:].reshape(dim,1)) - Y[j]) * 1

            else:

                grad += (hypothesis_func(parameter_arg, X[j,:].reshape(dim,1)) - Y[j]) * X[j,i-1]

        new_parameters.append(parameter_arg[i] - lr * grad)

    return np.array(new_parameters).reshape(dim+1,1)



X = np.array(range(10)).reshape(10,1)

Y = np.array(range(10)).reshape(10,1)



parameters = np.random.uniform(size=(2,1))



iter_number = []

losses = []



for i in range(1000):

    iter_number.append(i)

    losses.append(cost_func(parameters, X, Y)[0,0])

    parameters = GDStep(parameters, 0.001, X, Y)



print(parameters)

plot2DScatter(iter_number[-900:], losses[-900:], None, "Iterations", "Loss")
np_data = np.array(data)

np_data[:5]
X = np_data[:,:2]

Y = np_data[:,2]



parameters = np.random.uniform(size=(3,1))



iter_number = []

losses = []



for i in range(10000):

    per = 0

    iter_number.append(i)

    losses.append(cost_func(parameters, X, Y)[0,0])

    if i > 103:

        per = abs(losses[-2] - losses[-1])/abs(losses[100]-losses[101])

        if per < 0.01:

            break

    parameters = GDStep(parameters, 1e-9, X, Y)

#     print(hypothesis_func(parameters, X[0].reshape(2, 1)), losses[-1])

    print('\r', i, per, end='')

print()

print(parameters)

# plot2DScatter(iter_number, losses, None, "Iterations", "Loss")

plot2DScatter(iter_number[100:], losses[100:], None, "Iterations", "Loss")
for i in range(len(X)):

    print(Y[i], round(hypothesis_func(parameters, X[i].reshape(2, 1))[0,0], 2))