import matplotlib.pyplot as plt

import numpy as np



def myplot(x, y):

    ax = plt.gca()

    ax.spines['top'].set_color('none')

    ax.spines['right'].set_color('none')

    ax.spines['bottom'].set_position('zero')

    ax.spines['left'].set_position('zero')

    plt.plot(x, y)

    plt.show()
x = np.linspace(-5, 5, 501)

y = x * x - 2 * x + 1

myplot(x, y)
x = np.linspace(-5, 5, 501)

y = 2 ** x

myplot(x, y)
# sigmoid

x = np.linspace(-10, 10, 501)

y = 1 / (1 + np.exp(-x))

myplot(x, y)
x = np.linspace(-np.pi, np.pi, 51)

y = np.tan(x)

myplot(x, y)

# print(np.pi)