# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from numpy import arange, sin, cos, exp

from scipy.special import fresnel
from scipy import linspace
import matplotlib.pyplot as plt

      
i  = arange(5000)
x1 = 1.0*cos(i/10.0)*exp(-i/2500.0)
y1 = 1.4*sin(i/10.0)*exp(-i/2500.0)
d  = 450.0
vx = cos(i/d)*x1 - sin(i/d)*y1
vy = sin(i/d)*x1 + cos(i/d)*y1

plt.figure(1)            
plt.plot(vx, vy, "k")      
h = max(vy) - min(vy)
w = max(vx) - min(vx)
plt.axes().set_aspect(w/h)
plt.show()
