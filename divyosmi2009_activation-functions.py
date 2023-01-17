import matplotlib.pyplot as plt

import numpy as np

def sigmoid(x):

    return (1/(1+np.exp(-x)))
x = np.linspace(-10,10,100)

y = np.linspace(-10,10,10)
plt.plot(x,sigmoid(x),color = "orange",label = "x = linspace array -10,10,100")

plt.plot(y,sigmoid(y),color = "cyan",label = "y = linspace array -10,10,10")

plt.grid()

plt.title("sigmoid")

plt.legend()

plt.xlabel("X")

plt.ylabel("Y")
def softmax(x):

    e = np.exp(x - np.max(x)) 

    if e.ndim == 1:

        return e / np.sum(e, axis=0)

    else:  

        return e / np.array([np.sum(e, axis=1)]).T
plt.plot(x,softmax(x),color = "green",label = "x = linspace array -10,10,100")

plt.plot(y,softmax(y),color = "cyan",label = "y = linspace array -10,10,10")

plt.grid()

plt.title("softmax")

plt.legend()

plt.xlabel("X")

plt.ylabel("Y")
def ReLU(x):

    return x * (x > 0)
plt.plot(x,ReLU(x),color = "orange",label = "x = linspace array -10,10,100")

plt.plot(y,ReLU(y),color = "Red",label = "y = linspace array -10,10,10")

plt.grid()

plt.title("Relu")

plt.legend()

plt.xlabel("X")

plt.ylabel("Y")