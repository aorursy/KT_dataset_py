import matplotlib 

from matplotlib import pyplot as plt

from matplotlib import patches

import numpy as np

from numpy import heaviside as u

from ipywidgets import interact

import ipywidgets as widgets
def wave_operation(a, b, reverse_time = False, func = 'np.sin({})'):

    if reverse_time: 

        a *= -1

    N = 75 # number of points 

    x = np.linspace(-2*np.pi, 2*np.pi, N)

    y0 = eval(func.format('x')) # orignal function

    y1 = eval(func.format('a*x + b'))

    plt.plot(x, y0, label = func.format('t'))

    plt.plot(x, y1, label = func.format("{}*t+{}".format(a, round(b, 2))))

    plt.legend( loc='upper right', borderaxespad=0.)

    plt.title("Signal Translation Properties")

    plt.show()

    # add drop down to add different functions

    # input changes between at+b and a(t+b), show differences

    # add somehting about amplitude 

    # adding periodic signals, do this weekend. 

    # Lti visualiation 



functions = ['np.sin({})', 'np.cos({})', '({})**2', 'np.exp({})']

interact(wave_operation, a = (-1, 3, .1), b =(-np.pi/2, np.pi/2, np.pi/30), reverse_time = False, func = functions) #(min, max, step)
def step_functions(a0, b0, time_reverse0, a1, b1, time_reverse1, a2, b2, time_reverse2): # a is a list of multiplers while b is a list of adders, both len 5

    N = 50 # number of points

    l = 3 # number of step functions

    x = np.linspace(-5, 5, N)

    y = np.zeros(N)

    a = [a0, a1, a2]

    b = [b0, b1, b2]

    t_rev = [time_reverse0, time_reverse1, time_reverse2]

    k = [1]*l # k is used in time reversal 

    for i in range(l): 

        if t_rev[i]:

            k[i] = -1

    for i in range(len(y)):

        for j in range(len(a)): #O(n) (not avoidable)

            y[i] += a[j]*u(x[i]*k[j] + b[j], 0) # the u() function is a heavyside function(step func)

    plt.plot(x,y)

    plt.ylim((-3, 6))

    title = "y(t) = "

    for i in range(l-1):

        title += "{}*u(t + {}) + ".format(a[i], round(b[i], 3))

    title += "{}*u(t + {})".format(a[l-1], round(b[l-1], 3))

    plt.title(title)

    

step =.2

blims = (-5, 5, step) # (min, max, step)



interact(step_functions, a0 = widgets.FloatSlider(min=-3, max=3, step = step, value=1), b0 = blims, time_reverse0 = False,

         a1 = widgets.FloatSlider(min=-3, max=3, step= step, value=1), b1 = blims, time_reverse1 = False,

         a2 = widgets.FloatSlider(min=-3, max=3, step= step, value=1), b2  = blims, time_reverse2 = False)
def delta_approx(a, height):

    N = 400

    xlim = (-10, 10)

    x = np.linspace(xlim[0], xlim[1], N)

    y = height * np.exp(-(a*x)**2)

    dt = x[1] - x[0]

    A = sum(y)*dt

    plt.plot(x, y)

    plt.ylim(0, 50)

    plt.xlim(xlim)

    

    plt.title("Area under curve: {}".format(round(A, 2)))

    plt.show() 

    

interact(delta_approx, a = (0, 5, .2), height = (0, 50, .5) ) #(min, max, step)

    
def delta_and_step(epsilon):

    N = 100

    xlim = (-5, 5)

    t = np.linspace(xlim[0], xlim[1], N)

    delta = np.empty(N)

    dt = t[1] - t[0]

    step = np.empty(N)

    for i in range(N): #O(n)

        delta[i] = .5*(np.pi*epsilon)**-.5 * np.exp((-t[i]**2)/(4*epsilon))

        step[i] = sum(delta[0:i+1])*dt # numerical integral from delta(0 to i)

    fig, ax = plt.subplots(1,2, figsize=(20, 8))

    ax[0].plot(t, delta)

    ax[0].set_title(r"$\delta(t) = \frac {1}{2\sqrt{\pi*"+str(round(epsilon,3))+"}} e^{-t^2/(4*"+str(round(epsilon, 3))+ r")}$", pad = 20, fontsize = 'xx-large')

    ax[0].set_ylim(0, 5)

    ax[0].grid()

    ax[1].plot(t, step)

    ax[1].grid()

    ax[1].set_title(r"$\int_{-\infty}^{t} {\delta}(t) dt$", pad = 20, fontsize = 'xx-large')

    

interact(delta_and_step, epsilon = (.002, .2, .01))
print('\n/n')