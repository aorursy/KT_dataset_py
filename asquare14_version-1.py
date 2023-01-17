# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import csv

from pandas import read_csv

filename = '../input/road-electronic-city/10.csv'

data = read_csv(filename)

print(data.head())
x = data.iloc[:,0]

print(type(x))

print(x.head(10))

y = data.iloc[:,1]

print(type(y))

print(y.head(10))

z = data.iloc[:,2]

print(type(z))

print(z.head(10))



import matplotlib.pyplot as plt



x_accel = x

y_accel = y

z_accel = z



plt.subplot(3, 1, 1)

plt.plot(x_accel, '.-')

plt.title('A tale of 3 subplots')

plt.ylabel('X acceleration')



plt.subplot(3, 1, 2)

plt.plot(y_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Y acceleration')



plt.subplot(3, 1, 3)

plt.plot(z_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Z acceleration')



plt.show()
import csv

from pandas import read_csv

filename = '../input/road-electronic-city/3.csv'

data = read_csv(filename)

print(data.head())
x = data.iloc[:,0]

print(type(x))

print(x.head(10))

y = data.iloc[:,1]

print(type(y))

print(y.head(10))

z = data.iloc[:,2]

print(type(z))

print(z.head(10))



import matplotlib.pyplot as plt



x_accel = x

y_accel = y

z_accel = z



plt.subplot(3, 1, 1)

plt.plot(x_accel, '.-')

plt.title('A tale of 3 subplots')

plt.ylabel('X acceleration')



plt.subplot(3, 1, 2)

plt.plot(y_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Y acceleration')



plt.subplot(3, 1, 3)

plt.plot(z_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Z acceleration')



plt.show()
from matplotlib import pyplot

data.hist()

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
from pandas.plotting import scatter_matrix

scatter_matrix(data)

pyplot.show()
x = data.iloc[:,0]

print(type(x))

print(x.head(10))

y = data.iloc[:,1]

print(type(y))

print(y.head(10))

z = data.iloc[:,2]

print(type(z))

print(z.head(10))
import matplotlib.pyplot as plt



x_accel = x

y_accel = y

z_accel = z



plt.subplot(3, 1, 1)

plt.plot(x_accel, '.-')

plt.title('A tale of 3 subplots')

plt.ylabel('X acceleration')



plt.subplot(3, 1, 2)

plt.plot(y_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Y acceleration')



plt.subplot(3, 1, 3)

plt.plot(z_accel, '.-')

plt.xlabel('time (s)')

plt.ylabel('Z acceleration')



plt.show()