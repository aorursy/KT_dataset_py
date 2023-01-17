# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TIME_Q = 0.001
def X(time):
    return 5 if time > 1 else 1

Y = 1
k = 0.6
n = 10
betaY = 0.5
betaZ = 0.5
alphaZ = 0.5
alphaY = 0.5
Kyz = 3
Kxz = 2
def dY(X,Y):
    if X:
        return (betaY * X) - alphaY * Y
        return (betaY * k**n)/(k**n + Y**n) - alphaY * Y 
    else :
        return - alphaY * Y
def dZ(X,Y,Z):
    if X > Kxz and Y < Kyz:
        return betaZ - alphaZ * Z
    else :
        return - alphaZ * Z
Y_arr = []
Z_arr = []
Y = 1
Z = 0
for i in np.arange(0,10, TIME_Q):
    Y_arr.append(Y )
    Z_arr.append(Z)
    Y = Y + TIME_Q * dY(X(i), Y)
    Z = Z + TIME_Q * dZ(X(i), Y, Z)
fig, a = plt.subplots(3,1, sharex=True, figsize=(10,7))
a[0].plot([X(i) for i in np.arange(0,10, TIME_Q)], c="y")
a[1].plot(Y_arr, c="b")
a[1].plot([Kyz]*10000,"g-." , label="Kyz")
a[2].plot(Z_arr, c="r")

a[0].set_title("X")
a[1].set_title("Y")
a[2].set_title("Z")

a[1].axvline(np.argwhere(Z_arr)[0][0], color="grey",linestyle=":")
a[2].axvline(np.argwhere(Z_arr)[0][0], color="grey",linestyle=":")
a[1].legend()
np.argwhere(Z_arr)
