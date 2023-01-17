import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt


# function that returns dz/dt

def model(z,t):

    x = z[0]

    y = z[1]

    dxdt = 4*x-5*y+3

    dydt = 5*x-4*y+6

    dzdt = [dxdt,dydt]

    return dzdt



# f function

def f(Y, t):

    y1, y2 = Y

    return [4*y1-5*y2+3, 5*y1-4*y2+6]
y1 = np.linspace(-3, -1, 20)

y2 = np.linspace(-2, 0, 20)



Y1, Y2 = np.meshgrid(y1, y2)



t = 0



u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)



NI, NJ = Y1.shape



for i in range(NI):

    for j in range(NJ):

        x = Y1[i, j]

        y = Y2[i, j]

        yprime = f([x, y], t)

        u[i,j] = yprime[0]

        v[i,j] = yprime[1]

     



Q = plt.quiver(Y1, Y2, u, v, color='r')



plt.xlabel('$x$')

plt.ylabel('$y$')

plt.title('Orbits on phase space')

#plt.xlim([-2, 8])

#plt.ylim([-4, 4])
#Ploting solutions with different initial conditions



for x0 in [k * 0.01-2.1 for k in range(0, 20,2)]:

    for y0 in [w * 0.01-1.1 for w in range(0, 20,2)]:    

        # initial condition

        z0 = [x0,y0]



        # number of time points

        n = 400



        # time points

        t = np.linspace(0, 2,n)





        # store solution

        x = np.empty_like(t)

        y = np.empty_like(t)



        # record initial conditions

        x[0] = z0[0]

        y[0] = z0[1]



        # solve ODE

        for i in range(1,n):

            # span for next time step

            tspan = [t[i-1],t[i]]

            # solve for next step

            z = odeint(model,z0,tspan)

            # store solution for plotting

            x[i] = z[1][0]

            y[i] = z[1][1]

            # next initial condition

            z0 = z[1]





        #plot phase space

        plt.plot(x,y,'m',linewidth=.5)

        plt.ylabel('y')

        plt.xlabel('x')

        plt.axis('equal')

        plt.legend(loc='best')

        plt.title('Phase Space')



plt.grid()        

plt.show()