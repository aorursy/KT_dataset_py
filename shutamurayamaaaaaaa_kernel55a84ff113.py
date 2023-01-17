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
import matplotlib.pyplot as plt

import numpy as np

from mpl_toolkits.mplot3d import Axes3D

Xinit = 1.0

Yinit = 1.0

Zinit = 1.0



sigma = 10.0

r = 28.0

b = 8.0/3.0



def rk(dt,tend): #4th order Runge-Kutta Method

    t = np.arange(0.0,tend+dt,dt)

    X = np.zeros_like(t)

    Y = np.zeros_like(t)

    Z = np.zeros_like(t)

    X[0] = Xinit

    Y[0] = Yinit

    Z[0] = Zinit

    for i in range(0,t.shape[0]-1):

        h1 = dt*dXdt(t[i],X[i],Y[i],Z[i])

        j1 = dt*dYdt(t[i],X[i],Y[i],Z[i])

        k1 = dt*dZdt(t[i],X[i],Y[i],Z[i])

        h2 = dt*dXdt(t[i]+dt/2.,X[i]+h1/2.,Y[i]+j1/2.,Z[i]+k1/2.)

        j2 = dt*dYdt(t[i]+dt/2.,X[i]+h1/2.,Y[i]+j1/2.,Z[i]+k1/2.)

        k2 = dt*dZdt(t[i]+dt/2.,X[i]+h1/2.,Y[i]+j1/2.,Z[i]+k1/2.)

        h3 = dt*dXdt(t[i]+dt/2.,X[i]+h2/2.,Y[i]+j2/2.,Z[i]+k2/2.)

        j3 = dt*dYdt(t[i]+dt/2.,X[i]+h2/2.,Y[i]+j2/2.,Z[i]+k2/2.)

        k3 = dt*dZdt(t[i]+dt/2.,X[i]+h2/2.,Y[i]+j2/2.,Z[i]+k2/2.)

        h4 = dt*dXdt(t[i]+dt,X[i]+h3,Y[i]+j3,Z[i]+k3)

        j4 = dt*dYdt(t[i]+dt,X[i]+h3,Y[i]+j3,Z[i]+k3)

        k4 = dt*dZdt(t[i]+dt,X[i]+h3,Y[i]+j3,Z[i]+k3)

        X[i+1] = X[i] + (h1 + 2.0*h2 + 2.0*h3 + h4)/6.0

        Y[i+1] = Y[i] + (j1 + 2.0*j2 + 2.0*j3 + j4)/6.0

        Z[i+1] = Z[i] + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    return X,Y,Z,t



def dXdt(t,X,Y,Z):

    return (-sigma*X+sigma*Y)



def dYdt(t,X,Y,Z):

    return (-X*Z+r*X-Y)



def dZdt(t,X,Y,Z):

    return (X*Y-b*Z)



fig = plt.figure(figsize=(20,6))

X,Y,Z,t = rk(0.01,100.0)

plt.plot(t,X,label='X')

plt.plot(t,Y,label='Y')

plt.plot(t,Z,label='Z')

plt.legend()

plt.title('18B15129 Shuta Murayama')

plt.xlabel('time')

plt.ylabel('population density')

plt.show()



fig = plt.figure(figsize=(10,10))

ax = fig.gca(projection='3d')

surf = ax.scatter(X,Y,Z,c=t,s=1.0,cmap='gist_rainbow')

fig.colorbar(surf)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')

plt.title('18B15129 Shuta Murayama')

plt.show()
global Xinit

Xinit = 10.0

X1,Y2,Z3,t = rk(0.01,100.0)

fig = plt.figure(figsize=(10,10))

ax = fig.gca(projection='3d')

surf = ax.scatter(X1,Y2,Z3,c=t,s=1.0,cmap='gist_rainbow')

fig.colorbar(surf)

ax.set_xlabel('X')

ax.set_ylabel('Y')

ax.set_zlabel('Z')

plt.title('18B15129 Shuta Murayama')

plt.show()