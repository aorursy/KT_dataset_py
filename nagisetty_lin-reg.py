import numpy as np
import matplotlib.pyplot  as plt
#from scipy.interpolate import *
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
%matplotlib inline
x = np.arange(2,22,2)
y = np.arange(1,11,1)
x,y
yn = y + np.random.uniform(1,2,10)
yn = y +randn(10)

fit_model = polyfit(x,yn,1)
lin_reg = polyval(fit_model,x)
plt.plot(x,yn,'o')
plt.plot(x,lin_reg,'r-')
plt.show()