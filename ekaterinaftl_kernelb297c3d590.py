import matplotlib.pyplot as plt

import math

import pylab

from matplotlib import mlab



def func(t):

    return 2* math.cos(t-2) + math.sin(2*t-4)



# Интервал изменения переменной по оси X

tmin = -20*math.pi

tmax = 20*math.pi



# Шаг между точками

dt = 0.1



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (tmin, tmax, dt)



# Вычислим значение функции в заданных точках

ylist = [func (t) for t in xlist]



# !!! Нарисуем одномерный график с использованием стиля

pylab.plot (xlist, ylist, '-k' )



# !!! Покажем окно с нарисованным графиком



pylab.show()
import matplotlib.pyplot as plt

import math

import pylab

from matplotlib import mlab



def func(t):

    return 2* math.cos(t-2) + math.sin(2*t-4)



# Интервал изменения переменной по оси X

tmin = -20*math.pi

tmax = 20*math.pi



# Шаг между точками стал больше, график более угловат.

dt = 0.5



# !!! Создадим список координат по оси X на отрезке [-xmin; xmax], включая концы

xlist = mlab.frange (tmin, tmax, dt)



# Вычислим значение функции в заданных точках

ylist = [func (t) for t in xlist]



# !!! Нарисуем одномерный график с использованием стиля

pylab.plot (xlist, ylist, '--r' )



# !!! Покажем окно с нарисованным графиком



pylab.show()



plt.savefig('sonya.png') #сохранение графика