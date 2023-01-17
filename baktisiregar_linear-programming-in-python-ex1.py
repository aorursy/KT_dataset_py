import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Construct lines
x = np.linspace(0, 20, 2000)              # x > 0
y1 = (x*0) + 2                            # y >= 2
y2 = (25-x)/2.0                           # 2y <= 25 - x
y3 = (2*x-8)/4.0                          # 4y >= 2x - 8 
y4 = 2 * x -5                             # y <= 2x - 5 

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