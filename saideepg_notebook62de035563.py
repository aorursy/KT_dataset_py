# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sympy import symbols, init_printing

from sympy.plotting import plot

%matplotlib inline

init_printing()

x = symbols('x')

fx = x**4 - 3*x**3 + 2

p1 = plot(fx, (x, -2, 4), ylim=(-10,40)) #Plotting f(x) = x^4 - 3x^3 + 2, showing -2 < x <4