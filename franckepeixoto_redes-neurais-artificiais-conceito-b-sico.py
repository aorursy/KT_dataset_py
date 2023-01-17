import matplotlib.pyplot as plt

import seaborn; seaborn.set()

from numpy import exp, array, append

# Função de ativação Sigmoid



def Sigmoid(x):

    return 1 / (1 + exp(-x))



for i in array([range(-10, 10)]):

    x = append(array([]), i)

    plt.plot(x, Sigmoid(x))

    plt.title("Função de ativação Sigmoid")

    plt.xlabel("x")

    plt.ylabel("Sigmoid(x)")

# Tan-h 

def Tanh(x):

    return 2 / (1 + exp(-2*x)) - 1

for i in array([range(-10, 10)]):

    x = append(array([]), i)

    plt.plot(x, Tanh(x))

    plt.title("Função de ativação Tan-h")

    plt.xlabel("x")

    plt.ylabel("Tanh(x)")

    
from numpy import sum

# Softmax

def Softmax(x):

    return exp(x) / sum(exp(x))

x = array([1, 2, 3, 4, 5])

plt.plot(x, Softmax(x))

plt.title("Função de ativação Softmax")

plt.xlabel("x")

plt.ylabel("Softmax(x)")
# ReLU

def ReLU(x):

    return x * (x > 0)

x = array(range(-10, 10))

plt.plot(x, ReLU(x))

plt.title("Função de ativação ReLU")

plt.xlabel("x")

plt.ylabel("ReLU(x)")
from numpy import where

def LeakyReLU(x):

    return where(x > 0, x, x * 0.1)

x = array(range(-10, 10))

plt.plot(x, LeakyReLU(x))

plt.title("Função de ativação Leaky ReLU")

plt.xlabel("x")

plt.ylabel("Leaky ReLU(x)")