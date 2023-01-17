import numpy as np
import matplotlib.pyplot as plt

def eul(dt,t):
    T = np.copy(t)
    T[0] = 2.0
    for i in range(0,t.shape[0]-1):
        h1 = dt*dTdt(t[i],T[i])
        T[i+1] = T[i] + h1
    return T

def heun(dt,t):
    T = np.copy(t)
    T[0] = 2.0
    for i in range(0,t.shape[0]-1):
        h1 = dt*dTdt(t[i],T[i])
        h2 = dt*dTdt(t[i]+dt, T[i]+h1)
        T[i+1] = T[i] + 0.5*(h1+h2)
    return T

def rk4(dt,t):
    T = np.copy(t)
    T[0] = 2.0
    for i in range(0,t.shape[0]-1):
        h1 = dt*dTdt(t[i],T[i])
        h2 = dt*dTdt(t[i]+dt/2., T[i]+h1/2.)
        h3 = dt*dTdt(t[i]+dt/2., T[i]+h2/2.)
        h4 = dt*dTdt(t[i]+dt, T[i]+h3)
        T[i+1] = T[i] + (h1 + 2*h2 + 2*h3 + h4)/6.
    return T

def dTdt(t,T):
    return -5.*T

##Main Program
dt = 0.1
t = np.arange(0.,1.0+dt,dt)

T_exact = 2*np.exp(-5*t)
plt.plot(t,T_exact,'k-',label='Exact Solution')

T_eul = eul(dt,t)
plt.plot(t,T_eul,'b-',label='Euler Method')
T_heun = heun(dt,t)
plt.plot(t,T_heun,'g-',label='Heun Method')
T_rk4 = rk4(dt,t)
plt.plot(t,T_rk4,'r.',label='Runge Kutta 4th')
plt.legend()
plt.show()