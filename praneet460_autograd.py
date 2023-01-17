!pip install autograd
# Import required libraries

import autograd.numpy as np

from autograd import grad

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sn

style.use('ggplot')

%matplotlib inline
def f(x):

    # f(x) = x^3 + x^2 + 1

    result = np.power(x, 3) + np.power(x, 2) + 1

    return result

f(4.0)
# First order derivative

df = grad(f)

df(4.0)
# Second order derivative

d2f = grad(df)

d2f(4.0)
d3f = grad(d2f)

d3f(4.0)
from autograd import elementwise_grad as egrad

x = np.linspace(start = -7.0, stop = 7.0, num = 200)

y = []

first_dev = []

second_dev = []

third_dev = []

for i in range(0, len(x)):

    y.append(f(x[i]))

    first_dev.append(df(x[i]))

    second_dev.append(d2f(x[i]))

    third_dev.append(d3f(x[i]))

    







plt.plot(x, y, c='k', linestyle='--', label="x^3+x^2+1")

plt.plot(x, first_dev, c='blue', label="3x^2+2x")

plt.plot(x, second_dev, c='red', label="6x+2")

plt.plot(x, third_dev, c='grey', label="6")



plt.legend(loc="best")

plt.show()
from autograd import elementwise_grad as egrad # for functions that vectorize over inputs

def tanh(x):

    y = np.exp(-2 * x)

    return ((1 - y)/(1 + y))
# plot the graph

plt.figure(figsize=(14, 12))

plt.plot(x, tanh(x), label="tanh")

plt.plot(x, egrad(tanh)(x), label="first_derv")

plt.plot(x, egrad(egrad(tanh))(x), label="second_derv")

plt.plot(x, egrad(egrad(egrad(tanh)))(x), label="third_derv")

plt.legend(loc="best")

plt.show()