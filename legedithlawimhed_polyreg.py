# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
total_funds_used = [4500, 3797.1, 3418.8, 2876.4, 2662.8, 3374.5, 3938.8, 3533.1, 4903.1, 4183.1, 5014.3]
year = [2008,2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
no_of_fires = [78979, 78792, 71971, 74126, 67774, 47579, 63212, 68151, 67595, 71499, 58083]
acres_burnt = [5292468, 5921786, 3422724, 8711367, 9326238, 4319546, 3595613, 10125149, 5503538, 10026086, 8767492]

from sympy import S, symbols, printing
from matplotlib import pyplot as plt
import numpy as np
p = np.polyfit(year, no_of_fires, 2)
f = np.poly1d(p)
p = np.polyfit(year, total_funds_used, 5)
f = np.poly1d(p)
print(no_of_fires)
x_new = np.linspace(year[0], year[-1], 50)
y_new = f(x_new)
x = symbols("x")
poly = sum(S("{:6.2f}".format(v))*x**i for i, v in enumerate(p[::-1]))
eq_latex = printing.latex(poly)

plt.plot(x_new, y_new, label="${}$".format(eq_latex))
plt.plot(year, total_funds_used, 'o')

plt.legend(fontsize="small")
plt.show()
print(poly)
f(2020)