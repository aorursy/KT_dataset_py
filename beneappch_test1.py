%matplotlib inline

from ipywidgets import interactive

import matplotlib.pyplot as plt

import numpy as np



def f(m, b, t):

    plt.figure(2)

    x = np.linspace(-10, 10, num=1000)

    plt.plot(x, m * x + b)

    plt.ylim(-5, 5)

    plt.title(t)

    plt.show()



interactive_plot = interactive(f, m=(-2.0, 2.0), b=(-3, 3, 0.5), t='my title')

output = interactive_plot.children[-1]

output.layout.height = '350px'

interactive_plot
print('asdf')