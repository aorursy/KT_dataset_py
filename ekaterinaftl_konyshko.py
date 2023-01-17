# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import math

import pylab

from matplotlib import mlab



def func (x):

    return x**2



# Интервал изменения переменной по оси X

xmin = -2.0

xmax = 2.0



# Шаг между точками

dx = 0.1



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (xmin, xmax, dx)



# Вычислим значение функции в заданных точках

ylist = [func (x) for x in xlist]



# !!! Нарисуем одномерный график с использованием стиля

pylab.plot (xlist, ylist, '--r' )



# !!! Покажем окно с нарисованным графиком



pylab.show()
import matplotlib.pyplot as plt

import math

import pylab

from matplotlib import mlab



def func (x):

    return x**2



# Интервал изменения переменной по оси X

xmin = -2.0

xmax = 2.0



# Шаг между точками

dx = 0.6



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (xmin, xmax, dx)



# Вычислим значение функции в заданных точках

ylist = [func (x) for x in xlist]



# !!! Нарисуем одномерный график с использованием стиля

pylab.plot (xlist, ylist, '*g' )



# !!! Покажем окно с нарисованным графиком



pylab.show()

pylab.savefig('fig1.png')

import matplotlib.pyplot as plt

import math

import pylab

from matplotlib import mlab



def func (x):

    return math.sin(2*math.pi*x+0)



def func2 (x):

    return math.sin(2*math.pi*x+math.pi/6)



def func3 (x):

    return math.sin(2*math.pi*x+(2*math.pi)/6)



def func4 (x):

    return math.sin(2*math.pi*x+(3*math.pi)/6)



def func5 (x):

    return math.sin(2*math.pi*x+(4*math.pi)/6)



def func6 (x):

    return math.sin(2*math.pi*x+(5*math.pi)/6)



# Интервал изменения переменной по оси X

xmin = -1.0

xmax = 1.0



# Шаг между точками

dx = 0.1



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (xmin, xmax, dx)



# Вычислим значение функции в заданных точках



ylist = [func (x) for x in xlist]



ylist2 = [func2 (x) for x in xlist]



ylist3 = [func3 (x) for x in xlist]



ylist4 = [func4 (x) for x in xlist]



ylist5 = [func5 (x) for x in xlist]



ylist6 = [func6 (x) for x in xlist]





# !!! Нарисуем одномерный график с использованием стиля

pylab.plot (xlist, ylist, '--*g',label = '0')

pylab.plot (xlist, ylist2, '-*k',label = 'pi/6')

pylab.plot (xlist, ylist3, '--r',label = '2*pi/6')

pylab.plot (xlist, ylist, '--g',label = '3*pi/6')

pylab.plot (xlist, ylist2, '--y',label = '4*pi/6')

pylab.plot (xlist, ylist3, '*-c',label = '5*pi/6')





pylab.legend ()

# !!! Покажем окно с нарисованным графиком

pylab.grid()

pylab.show()
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import 

from matplotlib.colors import LinearSegmentedColormap 

import matplotlib.pyplot as plt 

import numpy as np 





x = np.arange(-3,3)



y = np.arange(-3,3)



X, Y = np.meshgrid(x, y)





Z = ((X) **2) + ((Y) **2)





fig1 = plt.figure()

p = fig1.add_subplot(111)

p.contourf(X,Y,Z) #построение контурной диаграммы

fig1.show()

fig1.savefig('fig4.png')



fig = plt.figure() 

# Построение 3D графика 

ax = plt.axes(projection="3d") 

ax.plot_surface(X, Y, Z) 

plt.savefig('3d.png')