import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import pylab

from matplotlib import mlab
math.log10(2*3-5)
import math

import pylab

from matplotlib import mlab



def func (x):

    try:

        return x**3 * math.sin(x)

    except:

        return None



# Интервал изменения переменной по оси X

xmin = -50.0

xmax = 50.0



# Шаг между точками

dx = 0.01



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (xmin, xmax, dx)



# Вычислим значение функции в заданных точках

ylist = [func (x) for x in xlist]



# !!! Нарисуем одномерный график

pylab.plot (xlist, ylist)



# !!! Покажем окно с нарисованным графиком

pylab.show()