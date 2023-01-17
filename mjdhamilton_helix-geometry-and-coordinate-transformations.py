# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#v2: moved most inline math expressions into equation environments
#v3: added discussion of cylindrical and spherical coordinates, corrected formulas involving charge sign
#v4: typos corrected
# Use time parameter s = omega*t

def x(s,R,phi_0):
    return R*(np.cos(phi_0-s)-np.cos(phi_0))

def y(s,R,phi_0):
    return R*(np.sin(phi_0-s)-np.sin(phi_0))

# p_L = p_longitudinal, p_T = p_transversal
def z(s,R,p_T,p_L):
    return R*(p_L/p_T)*s
def r1(s,R,phi_0,p_T,p_L):
    return np.sqrt(x(s,R,phi_0)**2+y(s,R,phi_0)**2+z(s,R,p_T,p_L)**2)

def r2(s,R,phi_0):
    return np.sqrt(x(s,R,phi_0)**2+y(s,R,phi_0)**2)
def x2(s,R,phi_0,p_T,p_L):
    return x(s,R,phi_0)/r1(s,R,phi_0,p_T,p_L)

def y2(s,R,phi_0,p_T,p_L):
    return y(s,R,phi_0)/r1(s,R,phi_0,p_T,p_L)

def z2(s,R,phi_0,p_T,p_L):
    return z(s,R,p_T,p_L)/r2(s,R,phi_0)
# Set some values for radius R and momenta p_L, p_T, used throughout the examples
R = 1
p_L = 100
p_T = 10
# Start with a short time interval [0.01, 0.5]
# We do not start in s=0 to avoid dividing by 0 when calculating x2, y2, z2
S = np.linspace(0.01, 0.5, 200)

# Plot (x,y) for 10 different values for phi_0, corresponding to different initial velocity vectors
for phi_0 in np.linspace(0, np.pi/2, 10):
      
    X = x(S,R,phi_0)
    Y = y(S,R,phi_0)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.plot(X,Y)
# Plot transformed coordinates (x2,y2) for the same angles
# The transformed curves align on a circle segment
for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)
# If we make the time interval very short, the different starting points, depending on phi_0, become obvious
S = np.linspace(0.01, 0.03, 200)

for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)
S = np.linspace(0.01, 0.5, 200)

# The circle becomes clearer if we let phi_0 run from 0 to 2*pi
for phi_0 in np.linspace(0, 2*np.pi, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)
# If the time interval becomes large, the segments do not align as well on the circle as before

S = np.linspace(0.01, 2, 200)

for phi_0 in np.linspace(0, np.pi/2, 10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)
for phi_0 in np.linspace(0,2*np.pi,10):
    
    X2 = x2(S,R,phi_0,p_T,p_L)
    Y2 = y2(S,R,phi_0,p_T,p_L)
    
    plt.axis("equal")
    plt.xlabel("x2")
    plt.ylabel("y2")
    
    plt.plot(X2,Y2)
# We now plot z2
# Back to the short time interval
S = np.linspace(0.01, 0.5, 200)

# z2 is independent of phi_0, set phi_0 = 0
phi_0 = 0

# Plot the time dependency of z2
# z2 is approximately constant and the quadratic time dependency is apparent
Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()
# With a larger time interval, higher order terms kick in
S = np.linspace(0.01, 5, 200)

phi_0 = 0

Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()
# The denominator of z2 becomes singular at integer multiples of 2*pi
S = np.linspace(0.01, 7, 200)

phi_0 = 0

Z2 = z2(S,R,phi_0,p_T,p_L)
plt.xlabel("s")
plt.ylabel("z2")
plt.plot(S,Z2)
plt.show()