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





def rk4(dt,t,Tinit):

    T = np.copy(t)

    T[0] = Tinit

    for i in range(0,t.shape[0]-1):

        h1 = dt*dTdt(t[i],T[i])

        h2 = dt*dTdt(t[i]+dt/2.,T[i]+h1/2.)

        h3 = dt*dTdt(t[i]+dt/2.,T[i]+h2/2.)

        h4 = dt*dTdt(t[i]+dt,T[i]+h3)

        T[i+1] = T[i]+(1.0/6.0)*(h1+2.*h2+2.*h3+h4)

    return T



def dTdt(t,T):

    return -5*T





##Main Program

dt = 0.1

t = np.arange(0.,1.0+dt,dt)

        

Tinit = 2.0

T_exact = 2*np.exp(-5*t)

plt.plot(t,T_exact,'ro')







T = rk4(dt,t,Tinit)

plt.plot(t,T)

plt.show()