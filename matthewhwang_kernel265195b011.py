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
import numpy as np
import matplotlib.pyplot as plt

dt = 0.1
t = np.arange(0.,1.0+dt,dt)
T_exact = 2*np.exp(-5*t)

plt.plot(t,T_exact,'k-',label = "exact solution")


#Euler method
def euler(dt,t):
    T = np.copy(t) # np.zeros will work too. just want same size
    T[0] =  2
    for i in range(0,t.shape[0]-1):
        h = dt*(T[i]+t[i])
        T[i+1] = T[i] + h
    return T 

def heun(dt,t):
    T = np.copy(t) # np.zeros will work too. just want same size
    T[0] =  2
    for i in range(0,t.shape[0]-1):
        h1 = dt*(T[i]+t[i])
        h2 = dt*dTdt(t[i]+dt,T[i]+h1)
        T[i+1] = T[i] + (h1 + h2)*0.5
    return T 

def runge(dt,t):
    T = np.copy(t) # np.zeros will work too. just want same size
    T[0] =  2
    for i in range(0,t.shape[0]-1):
        h1 = dt*(T[i]+t[i])
        h2 = dt*dTdt(t[i]+dt/2.0,T[i]+h1/2.0)
        h3 = dt*dTdt(t[i]+dt/2.0,T[i]+h2/2.0)
        h4 = dt*dTdt(t[i]+dt,T[i]+h3)
        T[i+1] = T[i] + (h1 + 2.0*h2 + 2.0*h3 + h4)*(1/6.0)
    return T 



def dTdt(t,T):
    return -5*T
T1 = euler(dt,t)
T2 = heun(dt,t)
T3 = runge(dt,t)
plt.plot(t,T1,'ro',label = 'euler')
plt.plot(t,T2,'go',label = "heun")
plt.plot(t,T3,'bo',label = "runge-kutta")
plt.legend()
plt.show()