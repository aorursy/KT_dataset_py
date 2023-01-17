%matplotlib inline
import numpy as np

import numpy.polynomial.polynomial as p



import sympy

from sympy.abc import x



import matplotlib.pyplot as plt



sympy.init_printing()
poly = [16, -10, -4, -24]
p.polyadd([1, 2], [3, 5, 0, 3])
result = p.polymul([5, 2.1, 0, 0, 9], [3.5, 5, 0, 3, 1])
#for i in range(len(result) - 1, -1, -1):

#    print(str(result[i]) + "*x^" + str(i), end = "+")

sympy.Poly(list(reversed(result)), x)
[x ** 2 for x in range(-5, 6)]
{x ** 2 for x in range(-5, 6)}
def to_plot(functions, x1, x2, points, grid = False, center = False, eqasp = False, labels = None):

    '''

    These optional arguments make this function to make it more powerful.

    grid - draws a grid to the plot

    center - centers the origin in the final image (this is a *very* expensive operation with lots of points/functions)

    eqasp - forces the aspect ratio of both axis to be 1

    labels - a list of strings which maps to the first array, gets displayed as a legend when plotting  

    '''

    

    ax = plt.gca()

    if grid:

        ax.spines['left'].set_position(('data', 0))

        ax.spines['bottom'].set_position(('data', 0))

        ax.spines['right'].set_position(('data', 0))

        ax.spines['top'].set_position(('data', 0))

        plt.grid()

    if eqasp: ax.set_aspect('equal')

            

    x = np.linspace(x1, x2, points)

    fs = [np.vectorize(f) for f in functions]

    ys = [f(x) for f in fs]

    

    if center:

        l = max(abs(x1), abs(x2), max([max(y) for y in ys]) + 1)

        plt.xlim(-l, l)

        plt.ylim(-l, l)



    i = 0

    for y in ys: 

        plt.plot(x, y, label = "" if labels == None else labels[i])

        i += 1

    if labels != None: plt.legend()

    plt.show()
to_plot([np.sin, np.cos], -10, 10, 1000, grid=True, labels = ["sin(x)", "cos(x)"])
phi = np.linspace(0, 3/4 * np.pi, 1000)

r = [1] * 1000



plt.polar(phi, r)
x = (3 + 0j) + (2 + 0j)

x
type(x)