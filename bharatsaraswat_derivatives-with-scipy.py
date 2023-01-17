from scipy.misc import derivative as deriv

import matplotlib.pyplot as plt 

import seaborn as sns

import numpy as np

sns.set(style="darkgrid")
def derivPlot(f):

    x = np.arange(-10, 10, 0.01)

    d = deriv(f, x)

    plt.plot(x,f(x),'g-') # green is function

    plt.plot(x,d,'r-') # red is derivative of the function

    plt.show()
f = lambda x: pow(x, 2)

derivPlot(f)
f = lambda x: pow(x, 3)

derivPlot(f)
f = lambda x: pow(x, 4)

derivPlot(f)
for i in range(2, 20):

    f = lambda x: pow(x, i)

    derivPlot(f)