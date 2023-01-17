# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
g = 10000 * np.random.random(101)

h = 10000 * np.random.random(101)
def Pl(L, x):

    P = []

    P.append(1)

    P.append(x)

    l = 2

    while l <= L:

        P.append((x * (2 * l - 1) * P[l-1] - (l - 1) * P[l-2]) / l)

        l = l + 1

    return P



def dPl(L, x):

    dP = []

    dP.append(0)

    dP.append(-math.sin(math.acos(x)))

    P = Pl(L, x)

    l = 2

    while l <= L:

        dP.append(((2 * l - 1) * (- math.sin(math.acos(x)) * P[l-1] + math.cos(x) * dP[l-1]) - (l - 1) * dP[l-2]) / l)

        l = l + 1

    return dP



def H_x(L, x):

    X = 0

    l = 1

    dP = dPl(L, x)

    while l <= 30:

        X = X + g[l] * dP[l]

        l = l + 1

    return X

    

    

    

def H_z(L, x):

    Z = 0

    l = 1

    P = Pl(L, x)

    while l <= L:

        Z = Z + g[l] * P[l]

        l = l + 1

    return Z



Hx = H_x(100, math.cos(52.2443 / 360 * 2 * math.pi))

Hz = H_z(100, math.cos(52.2443 / 360 * 2 * math.pi))

F = (Hx ** 2 + Hz ** 2) ** 0.5

print("H_x =" + str(Hx) + "nT")

print("H_z =" + str(Hz) + "nT")

print("F =" + str(F) + "nT")

print("Inklination =" + str(math.asin(Hx / F) / 2 / math.pi *360) + "°")
a = []

Q = 361

for i in range(Q):

    a.append((H_x(100, math.cos(i/Q * 2 * math.pi)) ** 2 + H_z(100, math.cos(i/Q * 2 * math.pi)) ** 2) ** 0.5)

plt.plot(a)