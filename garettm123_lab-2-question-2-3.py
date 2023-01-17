# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sympy import *

import sympy as sy

import math as m



#Part A plotting functions

x, y = symbols('x y')

r = 2

R = 4

h = 3

Fx = (h/(R-r))*(x-r)

Gx = h

p1 = plot(Fx, show=False)

p2 = plot(Gx, show=False)

p1.append(p2[0])

p1.show()
#Part B solving functions and integrating

Fy = Eq(Fx, y)

Fy = solve(Fy, x)#Returns list

Fy = Fy[0]#Take first (only) item in list

output = integrate(m.pi*(Fy**2),(y,0,Gx))

output
#Part C integrating for calculated range

A = solve(Fx)#Find x intercept of Fx

B = solve(Fx-Gx)#Find intercept of Fx and Gx

output = integrate(m.pi*((Fx-R)**2),(x,A,B))

output
#The geometrical signifigance is that we're calculating the area of cones.

#Part A is is making a cone centered around the Y axis

#Part B is making a cone centered around R
#Question 3

Sx = (1-x**2)/2

Sx_intercepts = solve(Sx)

A = Sx_intercepts[0]

B = Sx_intercepts[1]

output = integrate(m.pi*(Sx**2)/4,(x,A,B))

output