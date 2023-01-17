import numpy as np

import matplotlib.pyplot as plt



def eul(dt,t):

    T = np.copy(t)

    T[0] = 2.

    for i in range(0,t.shape[0]-1):

        h = dt*dTdt(t[i],T[i]) 

        T[i+1] = T[i] + h  

    return T





def rk(dt,t):

    T = np.copy(t)

    T[0] = Tinit

    for i in range(0,t.shape[0]-1):

        h1 = dt*dTdt(t[i],T[i])

        h2 = dt*dTdt(t[i]+dt,T[i]+h1)

        T[i+1] = T[i]+h1

    return T





def dTdt(t,T):

    return -5*T





##Main program

dt = 0.1

t = np.arange(0.,1.0+dt,dt)



# T'+5T=0, T(0)=2

# integral(-1/5T)dt = integral(1)dt

# (-1/5)ln(T) = t+C

# ln(T)=-5t+C

# T = e^(-5t+C) <- T(0)=2

# T = 2e^(-5t)



Tinit = 2.0

T_exact = np.exp(-5*t)*2

plt.plot(t,T_exact) #blue curve



T = eul(dt,t)

plt.plot(t,T) #orange curve



T = rk(dt,t)

plt.plot(t,T) #green curve

plt.show()



# The orange curve (Euler) and green curve (Runge Kutta) overlap (are the same) since the Euler

# method is the simplest first-order procedure of the Runge-Kutta method. 

# We can see that the estimation is quite close to the exact solution. 
