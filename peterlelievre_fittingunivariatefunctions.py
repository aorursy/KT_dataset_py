%matplotlib inline

# Import required packages:

import numpy as np

import matplotlib.pyplot as plt

import math
# Load and extract the data:

d = np.loadtxt('../input/dataunivariate/DataUnivariate.txt')

xd = d[:,0]

yd = d[:,1]
fig = plt.figure()

# Plot the data as points, with a specific marker size and color:

plt.scatter(xd,yd, color='green')

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('f(x)')

# Set figure background color white:

fig.patch.set_facecolor('white')
# Define the univariate function to be plotted:

def f(x):

    return 1-0.4*x

# (change the line immediately above if you want to change the function)
# Define the x data points at which to plot the function:

dx = 0.05 # point spacing

xmin = 0 # minimum x value

xmax = 3 # maximum x value
# Calculate the x data points at which to plot the function:

xp = np.arange(xmin,xmax+dx,dx)

# ( x=np.arange(A,B,D) defines the x data points as points spaced D apart from A and B )

# "Vectorize" the function:

f2 = np.vectorize(f)

# Calculate the function values:

yp = f2(xp)

# Plot the data as points, with a specific marker size and color:

fig = plt.figure()

plt.scatter(xd,yd, color='green')

# Plot the function f(x) with a specific line width (lw) and color:

plt.plot(xp,yp, lw=2, color='orange')

# Add labels on the axes:

plt.xlabel('x')

plt.ylabel('f(x)')

# Set figure background color white:

fig.patch.set_facecolor('white')