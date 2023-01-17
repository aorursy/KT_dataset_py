import math

import matplotlib.pyplot as plt

import numpy as np
def sigmoid(z):

    return (1/(1 + np.exp(-z)))
print("sigmoid of a positive number:", sigmoid(1))

print("sigmoid of a large positive number:", sigmoid(100000))

print("sigmoid of a negative number:", sigmoid(-5))

print("sigmoid of a large negative number:", sigmoid(-100))
def line_graph(x, y, x_title, y_title):

    plt.plot(x, y)

    plt.xlabel(x_title)

    plt.ylabel(y_title)

    plt.show()
x = np.arange(-5, 6)

y = sigmoid(x)
print(x)

print(y)
line_graph(x, y, "Inputs", "Outputs")