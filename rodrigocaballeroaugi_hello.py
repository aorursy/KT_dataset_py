from __future__ import print_function

import numpy as np

from matplotlib import pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
# adimensional form

def SIR(x,t,Ro,d):

    # x[0] susceptible

    # x[1] infected

    # x[2] recovered, immune

    # x[3] population

    S,I,R,P = x

    xdot = np.array([-Ro*S*I/P,

                      Ro*S*I/P - (1.+d)*I,

                      I,

                     -d*I])

    return xdot



from scipy.integrate import odeint

def solve(N,L,x0,Ro,r,d):

    t = np.linspace(0,L,N)*r

    x = odeint(SIR, x0, t, args=(Ro,d))

    # output

    x = x*100. # convert to %

    return x.T
# parameters

Ro = 3.

r = 1./14. # recovery rate (1/illness duration, unit days-1)

d = .01   # death rate

N = 100  # no. timesteps

L = 300.   # length run (days)



# initial conditions

x0 = np.array([0.999, 0.001, 0., 1.])

S,I,R,P = solve(N, L, x0, Ro, r, d/100.)
%matplotlib inline

time = np.linspace(0,L,N)

fig,ax = plt.subplots(1,1)

l1,=ax.plot(time,S)

l2,=ax.plot(time,I)

l3,=ax.plot(time,R)

l4,=ax.plot(time,100-P,'k-')

ax.set_xlabel('days')

ax.set_ylabel('% of population')

ax.legend(['S','I','R','deaths'])



@interact(Ro=widgets.FloatSlider(min=0.5, max=6, step=0.25, value=2,continuous_update=True),

          d=widgets.IntSlider(min=0, max=100, step=5, value=2,continuous_update=True))



def make_plot(Ro,d):

    # integrate

    S,I,R,P = solve(N, L, x0, Ro, r, d/100.)

    l1.set_ydata(S)

    l2.set_ydata(I)

    l3.set_ydata(R)

    l4.set_ydata(100-P)

    fig.canvas.draw()

    fig.canvas.flush_events()