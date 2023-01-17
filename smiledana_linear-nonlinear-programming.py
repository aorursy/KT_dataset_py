# Import some other libraries that need
import matplotlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Design variables at mesh points
x = np.arange(-5.0, 10.0, 0.02)
y = np.arange(-5.0, 10.0, 0.02)
x1, x2 = np.meshgrid(x, y)

# Equations and Constraints
z = x2-8/x1
y1 = 0.2*x1-x2
y2 = 16-(x1-5)**2-x2**2

# Create a contour plot
plt.figure()
# Weight contours
lines = np.linspace(0.0,3.0,5)
CS = plt.contour(x1,x2,z,lines,colors='g')
plt.clabel(CS, inline=1, fontsize=10)

# y1
CS1 = plt.contour(x1,x2,y1,[0.0],colors='r')
# y2
CS2 = plt.contour(x1,x2,y2,[0.0],colors='b')


A = CS1.collections[0].get_paths()[0].vertices
B = CS2.collections[0].get_paths()[0].vertices
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

x = np.linspace(0, 20, 2000)
# y >= 2
y1 = (x*0) + 2
# 2y <= 25 - x
y2 = (25-x)/2.0
# 4y >= 2x - 8 
y3 = (2*x-8)/4.0
# y <= 2x - 5 
y4 = 2 * x -5

# Make plot
plt.plot(x, y1, label=r'$y\geq2$')
plt.plot(x, y2, label=r'$2y\leq25-x$')
plt.plot(x, y3, label=r'$4y\geq 2x - 8$')
plt.plot(x, y4, label=r'$y\leq 2x-5$')
plt.xlim((0, 16))
plt.ylim((0, 11))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

# Fill feasible region
y5 = np.minimum(y2, y4)
y6 = np.maximum(y1, y3)
plt.fill_between(x, y5, y6, where=y5>y6, color='grey', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)