import numpy as np

import matplotlib.pyplot as plt



def rk4(dt,t,Tinit):

    T = np.copy(t)

    T[0] = Tinit

    for i in range(0,t.shape[0]-1):

        h1 = dt*dTdt(t[i],T[i])

        h2 = dt*dTdt(t[i]+dt/2.,T[i]+(h1/2))

        h3 = dt*dTdt(t[i]+dt/2.,T[i]+(h2/2))

        h4 = dt*dTdt(t[i]+dt,T[i]+h3)

        T[i+1] = T[i]+((1/6)*(h1+(2*h2)+(2*h3)+h4))

    return T



def dTdt(t,T):

    return -5*T



## Main Program

dt = 0.1

t = np.arange(0.,1.0+dt,dt)



Tinit = 2.0

T_exact = 2*np.exp(-5*t)

plt.plot(t,T_exact,'ro')



T = rk4(dt,t,Tinit)

plt.plot(t,T)

plt.show()
