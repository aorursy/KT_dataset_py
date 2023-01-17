# Define libraries

import numpy as np

import pylab

from pylab import *
#Plot exercise 8.4-4



xvalues, yvalues = meshgrid(arange(-6, 6, 0.1), arange(-6, 6, 0.1))

xdot = -6*yvalues+2*xvalues*yvalues-8

ydot = yvalues**2-xvalues**2

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot exercise 8.4-6



xvalues, yvalues = meshgrid(arange(-1, 1, 0.1), arange(-2, 2, 0.1))

xdot = -yvalues*np.sqrt(1-xvalues**2)

ydot = xvalues*np.sqrt(1-xvalues**2)

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot exercise 8.4-8



xvalues, yvalues = meshgrid(arange(-10, 10, 0.1), arange(-2, 2, 0.1))

xdot = np.sin(xvalues+yvalues)

ydot = yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot exercise 8.4-13



xvalues, yvalues = meshgrid(arange(-1, 5, 0.1), arange(-2, 2, 0.1))

xdot = xvalues**2+3*xvalues*yvalues-4*xvalues

ydot = 2*xvalues*yvalues-6*yvalues**2+4*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot exercise 8.4-16



xvalues, yvalues = meshgrid(arange(-2, 2, 0.1), arange(-2, 2, 0.1))

xdot = -yvalues+(xvalues-yvalues)*(xvalues**2+yvalues**2-1)

ydot = xvalues+(xvalues+yvalues)*(xvalues**2+yvalues**2-1)

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Plot exercise 8.4-17



mu=10

xvalues, yvalues = meshgrid(arange(-2, 2, 0.1), arange(-2, 2, 0.1))

xdot = yvalues

ydot = -xvalues-mu*(xvalues**2-1)*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()



mu=-10

xvalues, yvalues = meshgrid(arange(-2, 2, 0.1), arange(-2, 2, 0.1))

xdot = yvalues

ydot = -xvalues-mu*(xvalues**2-1)*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
#Lotka Voltera

a=1;b=2;c=4;d=2

xvalues, yvalues = meshgrid(arange(-1, 4, 0.1), arange(-1, 1, 0.1))

xdot = (a-b*yvalues)*xvalues

ydot = (-c+d*xvalues)*yvalues

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()