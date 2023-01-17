from numpy import *

import math

import matplotlib.pyplot as plt

import numpy as np
t = linspace(-4, 2, 400)

a = t**2+1

b = np.exp(2*t)

c = a - b



plt.plot(t, a, 'r-',label='x^2+1') # plotting t, a separately 

plt.plot(t, b, 'b-',label='e^(2x)') # plotting t, b separately 

plt.xlabel('x')

plt.ylabel('functions')

plt.grid()

plt.legend(loc='best')

plt.show()



plt.plot(t, c, 'g-',label='x^2+1-e^(2x)') # plotting t, c separately 

plt.xlabel('x')

plt.ylabel('functions')

plt.grid()

plt.legend(loc='best')

plt.show()