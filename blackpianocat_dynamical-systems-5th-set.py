import numpy as np

import matplotlib.pyplot as plt

import pylab



from pylab import *

from mpl_toolkits import mplot3d
def f1(mu):

    return np.sqrt(1-np.sqrt(mu))

def f2(mu):

    return np.sqrt(1-np.sqrt(mu)/2)

def f3(mu):

    return np.sqrt(1+np.sqrt(mu)/2)

def f4(mu):

    return np.sqrt(1+np.sqrt(mu))



t1 = np.arange(0, 20, 0.1)

t2=np.arange(-2,0,0.1)



plt.plot(t1,f1(t1),'r--',t1,f2(t1),'b-',t1,f3(t1),'g--',t1,f4(t1),'m-',t1,0*t1,'k-')

plt.plot(t2,0*t2,'k--')

plt.xlabel('$\mu$')

plt.ylabel('r')

plt.show()
def f1(mu):

    return 1-np.sqrt(mu)

def f2(mu):

    return 1+np.sqrt(mu)



t1 = np.arange(-1, 1, 0.01)

t2= np.arange(1,5,0.01)



plt.plot(t1,f1(t1),'r--',t1,f2(t1),'b-',t1,t1*0,'k-')

plt.plot(t2,f2(t2),'b-',t2,t2*0,'k--')

plt.xlabel('$\mu$')

plt.ylabel('r')

plt.show()
%matplotlib inline

import numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os

from scipy.integrate import odeint

from mpl_toolkits.mplot3d.axes3d import Axes3D
font_family = 'Myriad Pro'

title_font = fm.FontProperties(family=font_family, style='normal', size=20, weight='normal', stretch='normal')
def lorentz_plotter(rh):

    

    fig = plt.figure(figsize=(12, 9))

    

    

    for i in np.arange(-3,5,2):

        # define the initial system state (aka x, y, z positions in space)

        initial_state = [i, 0, 0]



        # define the time points to solve for, evenly spaced between the start and end times

        start_time = 0

        end_time = 10

        time_points = np.linspace(start_time, end_time,10**4)

        

        # define the system parameters sigma, rho, and beta

        sigma = 10.

        rho = rh

        beta  = 8./3.

        

        # define the lorenz system

        # x, y, and z make up the system state, t is time, and sigma, rho, beta are the system parameters

        def lorenz_system(current_state, t):

    

            # positions of x, y, z in space at the current time point

            x, y, z = current_state



            # define the 3 ordinary differential equations known as the lorenz equations

            dx_dt = sigma * (y - x)

            dy_dt = x * (rho - z) - y

            dz_dt = x * y - beta * z



            # return a list of the equations that describe the system

            return [dx_dt, dy_dt, dz_dt]



        # use odeint() to solve a system of ordinary differential equations

        # the arguments are: 

        # 1, a function - computes the derivatives

        # 2, a vector of initial system conditions (aka x, y, z positions in space)

        # 3, a sequence of time points to solve for

        # returns an array of x, y, and z value arrays for each time point, with the initial values in the first row

        xyz = odeint(lorenz_system, initial_state, time_points)



        # extract the individual arrays of x, y, and z values from the array of arrays

        x = xyz[:, 0]

        y = xyz[:, 1]

        z = xyz[:, 2]



        # plot the lorenz attractor in three-dimensional phase space

        ax = fig.gca(projection='3d')

        ax.xaxis.set_pane_color((1,1,1,1))

        ax.yaxis.set_pane_color((1,1,1,1))

        ax.zaxis.set_pane_color((1,1,1,1))

        ax.plot(x, y, z, alpha=0.7, linewidth=0.6,label='$x(0)={i}$'.format(i=i))

        ax.plot(x, z, 'r--', zdir='y', zs=400,linewidth=0.3)

        

    plt.title('Lorentz attractor for ρ=%d' %rho,fontproperties=title_font)

    plt.legend(loc='best')

    plt.show()
for j in np.arange(170,138,-2):

    lorentz_plotter(j)
%matplotlib inline

from ipywidgets import interact, interactive

from IPython.display import clear_output, display, HTML



import numpy as np

from scipy import integrate



from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import cnames

from matplotlib import animation
def solve_lorenz(N=10, angle=0.0, max_time=4.0, sigma=10.0, beta=8./3, rho=28.0):



    fig = plt.figure()

    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    ax.axis('off')



    # prepare the axes limits

    ax.set_xlim((-25, 25))

    ax.set_ylim((-35, 35))

    ax.set_zlim((5, 55))



    def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho):

        """Compute the time-derivative of a Lorenz system."""

        x, y, z = x_y_z

        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]



    # Choose random starting points, uniformly distributed from -15 to 15

    np.random.seed(1)

    x0 = -15 + 30 * np.random.random((N, 3))



    # Solve for the trajectories

    t = np.linspace(0, max_time, int(250*max_time))

    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0i, t)

                      for x0i in x0])



    # choose a different color for each trajectory

    colors = plt.cm.viridis(np.linspace(0, 1, N))



    for i in range(N):

        x, y, z = x_t[i,:,:].T

        lines = ax.plot(x, y, z, '-', c=colors[i])

        plt.setp(lines, linewidth=2)



    ax.view_init(30, angle)

    plt.show()



    return t, x_t
w = interactive(solve_lorenz, angle=(0.,360.), max_time=(0.1, 4.0),

                N=(0,50), sigma=(0.0,50.0), rho=(0.0,200.0))

display(w)