# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df2 = pd.read_csv("../input/UNIVARIATE polynomial REG.csv")

plt.scatter(df2['X'], df2['Y'])
m_tot = df2['X'].count()
#****************
teta0_v = -25
learning_rate_teta0 = 1.5
learning_rate_teta1 = 1.5
#****************
Y_pred = []
Error = []
teta1 = []
teta0 = []
derivative_teta1 = []
derivative_teta0 = []

# loop teta0
for j in range(0, 50):
  #*****************
  teta1_v = -25
  #*****************
  
  # loop teta1
  for i in range(0, 50):
    Y_pred = []
    
    # loop Xi
    for m in range(0, m_tot):
      Y_pred.append(np.sqrt(df2['X'][m])*teta0_v + (df2['X'][m])**7*teta1_v)

    df2['Y_pred'] = Y_pred
    df2['SE'] = (df2['Y_pred'] - df2['Y'])**2
    df2['SE_derivative_teta1'] = (df2['Y_pred'] - df2['Y'])*df2['X']
    df2['SE_derivative_teta0'] = (df2['Y_pred'] - df2['Y'])
  #  plt.figure()
    plt.scatter(df2['X'], df2['Y'])
    plt.scatter(df2['X'], df2['Y_pred'])
    Error.append(sum(df2['SE'])/(2*m_tot))
    derivative_teta1.append(sum(df2['SE_derivative_teta1'])/(m_tot))
    derivative_teta0.append(sum(df2['SE_derivative_teta0'])/(m_tot))
    teta0.append(teta0_v)
    teta1.append(teta1_v)
    teta1_v = teta1_v + learning_rate_teta1
  
  teta0_v = teta0_v + learning_rate_teta0

J = pd.DataFrame()
J['teta1'] = teta1
J['teta0'] = teta0
J['error'] = Error
J['derivative_teta1'] = derivative_teta1
J['derivative_teta0'] = derivative_teta0
plt.figure()
plt.plot(J['teta1'], J['error'])
plt.plot(J['teta0'], J['error'])
J.round(2)
plt.figure()
plt.plot(J['teta1'], J['derivative_teta1'])
plt.plot(J['teta0'], J['derivative_teta0'])
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ax1.scatter(J['teta1'],J['teta0'],J['error'])
plt.xlabel('teta1', fontsize=16)
plt.ylabel('error', fontsize=16)
plt.ylabel('teta0', fontsize=16)
ax1.view_init(30, 30)

# rotate the axes and update
#for angle in range(0, 360):
#    ax1.view_init(30, angle)
#    plt.draw()
#    plt.pause(.001)
import scipy.interpolate

N = 500 #number of points for plotting/interpolation    
x = J['teta0']
y = J['teta1']
z = J['error']
xll = x.min();  xul = x.max();  yll = y.min();  yul = y.max()

xi = np.linspace(xll, xul, N)
yi = np.linspace(yll, yul, N)
zi = scipy.interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

contours = plt.contour(xi, yi, zi, 6, colors='white')
plt.clabel(contours, inline=True, fontsize=13)
plt.imshow(zi, extent=[xll, xul, yll, yul], origin='lower', cmap=plt.cm.jet, alpha=0.9)
plt.xlabel(r'$teta0$')
plt.ylabel(r'$teta1$')
plt.clim(0, 10)
plt.colorbar()
plt.show()
Xarray = np.array([np.sqrt(df2['X']), df2['X']**7])
yarray = np.array([df2['Y']])

X = np.mat(Xarray).T
y = np.mat(yarray).T

#**********************************NORMAL EQUATION
from numpy.linalg import inv

A = X.T.dot(X)
Ainv = inv(A)
THETA = Ainv.dot(X.T).dot(y)

param = pd.DataFrame(THETA)
param.head()