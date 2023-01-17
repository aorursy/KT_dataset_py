# Define libraries

import numpy as np

import pylab

from pylab import *
#Plot gradient system phase portraits



xvalues, yvalues = meshgrid(arange(-8, 8, 0.1), arange(-3, 3, 0.1))

xdot = -2*xvalues

ydot = -4*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot gradient system phase portraits



xvalues, yvalues = meshgrid(arange(-8, 8, 0.1), arange(-3, 3, 0.1))

xdot = -2*xvalues

ydot = 2*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot gradient system phase portraits



xvalues, yvalues = meshgrid(arange(-8, 8, 0.1), arange(-3, 3, 0.1))

xdot = -yvalues*np.cos(xvalues)

ydot = -np.sin(xvalues)

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()