import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
x = np.loadtxt('../input/radiance.rax', usecols = (0,))

y = np.loadtxt('../input/radiance.rax', usecols = (1,))
plt.plot(x,y,linewidth = 1.)
#How many lines do we have?

x.shape
def Planck(x,T):

    c1 = 119090000.

    c2 = 14387.9

    return (c1/x**5)*(1./(np.exp(c2/(x*T))-1.))
rows=1589

T0 = np.array([300.])  #we choose a T0 to start the optimization (that would be 300K)

sigma = np.ones(rows)

T0,sigma
import scipy.optimize as optimization

optimization.curve_fit(Planck,x,y,T0,sigma)
Kelvin = 273.15

Tfit =  303.32892516

Tc = Tfit - Kelvin

Tc
plt.plot(x,y)

plt.plot(x,Planck(x,Tfit).astype(np.float))

plt.legend(('Data','Fit'),

           loc='upper right', shadow=True)

plt.xlabel('Wavelength (um)')

plt.ylabel('Radiance (W m-2 sr-1)')

plt.show()