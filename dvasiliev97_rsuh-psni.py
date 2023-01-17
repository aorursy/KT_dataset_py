import numpy as np # linear algebra

import pandas as pd

import math

from scipy.optimize import fsolve

from matplotlib import pyplot as plt
print(np.roots([2, -13, -4, 60]))
print(np.roots([10, -30, 10]))
print(np.roots([1, -1, -15, 1, 38, 24]))
np.roots([16,0,-20,0,5,-1])
np.roots([32,0,-48,0,18,0,-1])
from sympy import *

x = symbols('x')

init_printing(use_unicode=True)
solveset(sin(x) - x + (1/6)*x**3, x)
solveset(sin(x) - 2*x - (1/6)*x**3, x)
f = lambda x : math.sin(x) - x + (1/6)*x**3

fsolve(f, 0.0001)
g = lambda x : math.sin(x) - 2*x - (1/6)*x**3

fsolve(g, 0.0001)