import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
x = np.arange(-5, 5, 0.01)
def plot(func, yaxis=(-1.4, 1.4)):
    plt.ylim(yaxis)
    plt.locator_params(nbins=5)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.axhline(lw=1, c='black')
    plt.axvline(lw=1, c='black')
    plt.grid(alpha=0.4, ls='-.')
    plt.box(on=None)
    plt.plot(x, func(x), c='r', lw=3)
binary_step = np.vectorize(lambda x: 1 if x > 0 else 0, otypes=[np.float])
plot(binary_step, yaxis=(-0.4, 1.4))
piecewise_linear = np.vectorize(lambda x: 1 if x > 3 else 0 if x < -3 else 1/6*x+1/2, otypes=[np.float])
plot(piecewise_linear, yaxis=(-0.4, 1.4))
bipolar = np.vectorize(lambda x: 1 if x > 0 else -1, otypes=[np.float])
plot(bipolar)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
plot(sigmoid, yaxis=(-0.4, 1.4))
def bipolar_sigmoid(x):
    return (1 - np.exp(-x)) / (1 + np.exp(-x))
plot(bipolar_sigmoid)
def tanh(x):
    return 2 / (1 + np.exp(-2 * x)) -1
plot(tanh)
def arctan(x):
    return np.arctan(x)
plot(arctan)
relu = np.vectorize(lambda x: x if x > 0 else 0, otypes=[np.float])
plot(relu, yaxis=(-0.4, 1.4))
leaky_relu = np.vectorize(lambda x: max(0.1 * x, x), otypes=[np.float])
plot(leaky_relu)
elu = np.vectorize(lambda x: x if x > 0 else 0.5 * (np.exp(x) - 1), otypes=[np.float])
plot(elu)
def softplus(x):
    return np.log(1+np.exp(x))
plot(softplus, yaxis=(-0.4, 1.4))