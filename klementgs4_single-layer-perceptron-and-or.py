%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import optimize
from ipywidgets import *
from IPython.display import SVG
from sklearn import datasets

AND = pd.DataFrame({'x1': (0,0,1,1), 'x2': (0,1,0,1), 'y': (0,0,0,1)})
AND
w = np.random.randn(3)*1e-4
inputs = AND[['x1','x2']]
target = AND['y']
inputs

g = lambda inputs, weights: np.where(np.dot(inputs, weights)>0, 1, 0)
def train(inputs, targets, weights, eta, n_iterations):

    # Add the inputs that match the bias node
    inputs = np.c_[inputs, -np.ones((len(inputs), 1))]

    for n in range(n_iterations):

        activations = g(inputs, weights);
        print(n,activations)
        weights -= eta*np.dot(np.transpose(inputs), activations - targets)
        print(n,weights)
        
    return(weights)
w = train(inputs, target, w, 0.25, 10)
w
-np.ones((len(inputs), 1))
a=np.c_[inputs, -np.ones((len(inputs), 1))]
a
g(a,w)
OR = pd.DataFrame({'x1': (0,0,1,1), 'x2': (0,1,0,1), 'y': (0,1,1,1)})
OR
w = np.random.randn(3)*1e-4
inputs = OR[['x1','x2']]
target = OR['y']
w = train(inputs, target, w, 0.25, 20)
g(np.c_[inputs, -np.ones((len(inputs), 1))], w)
AND.plot(kind='scatter', x='x1', y='x2', c='y', s=50, colormap='winter')
plt.plot(np.linspace(0,1.4), 1.5 - 1*np.linspace(0,1.4), 'k--');
OR.plot(kind='scatter', x='x1', y='x2', c='y', s=50, colormap='winter')
plt.plot(np.linspace(-.4,1), .5 - 1*np.linspace(-.4,1), 'k--');
XOR = pd.DataFrame({'x1': (0,0,1,1), 'x2': (0,1,0,1), 'y': (0,1,1,0)})

XOR.plot(kind='scatter', x='x1', y='x2', c='y', s=50, colormap='winter');