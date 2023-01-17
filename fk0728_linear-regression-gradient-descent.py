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
# Additional imports

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import animation

plt.style.use('seaborn')
# Real values

x1 = np.random.uniform(-50, 50, 1000)

x2 = np.random.uniform(-50, 50, 1000)



y_ideal = 5*x1 - 4*x2 + 8



noise = np.random.normal(0, 30, 1000)

y_true = y_ideal - noise



df = pd.DataFrame.from_dict({'x1':x1, 'x2':x2, 'y_true':y_true, 'y_ideal':y_ideal})
weights = np.array([-15, 20])

bias = 5

eta = 0.0000001
def predicted(_data: np.ndarray, _weights: np.ndarray, _bias: float) -> np.ndarray:

    return np.dot(_weights.T, _data.values.T) + _bias
def mean_square_error(_y_true: np.ndarray, _data: np.ndarray, _weights: np.ndarray, _bias: float) -> float:

    _predicted = predicted(_data, _weights, _bias)

    errors = (_y_true - _predicted)**2

    return errors.mean()
def derivative_weights(_y_true: np.ndarray, _data: np.ndarray, _weights: np.ndarray, _bias: float) -> np.ndarray:

    _predicted = predicted(_data, _weights, _bias)

    return -2*np.dot((_y_true - _predicted), _data)



def derivative_bias(_y_true: np.ndarray, _data: np.ndarray, _weights: np.ndarray, _bias: float) -> float:

    _predicted = predicted(_data, _weights, _bias)

    return -2*sum(_y_true - _predicted)
list_mse = []

list_weights = []

list_bias = []

for i in range(100):

    weights = weights - eta*derivative_weights(df['y_true'], df[['x1', 'x2']], weights, bias)

    bias = bias - eta*derivative_bias(df['y_true'], df[['x1', 'x2']], weights, bias)

    list_weights.append(weights)

    list_bias.append(bias)

    list_mse.append(mean_square_error(df['y_true'], df[['x1', 'x2']], weights, bias))
plt.plot(list_mse)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# ax.scatter(df.x1, df.x2, df.y_true, alpha=0.2, c='red')

p = []
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.x1, df.x2, df.y_true, alpha=0.2, c='red')



new_df = predicted(df[['x1', 'x2']], list_weights[-1], bias)

ax.scatter(df.x1, df.x2, new_df, alpha=0.5)

plt.pause(0.001)



plt.show()