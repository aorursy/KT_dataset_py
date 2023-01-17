import numpy as np
import matplotlib.pyplot as plt

def rk4(dt,t,Tinit=2): #Tinit is 2 by default. It can be changed by assigning Tinit on call
    T = np.copy(t)
    T[0] = Tinit
    for i in range(0,t.shape[0]-1):
        h1 = dt*dTdt(t[i],T[i])
        h2 = dt*dTdt(t[i]+dt/2,T[i]+h1/2)
        h3 = dt*dTdt(t[i]+dt/2,T[i]+h2/2)
        h4 = dt*dTdt(t[i]+dt,T[i]+h3)
        T[i+1] = T[i]+(1/6)*(h1+2*h2+2*h3+h4)
    return T

def eul(dt,t,Tinit=2): #Tinit is 2 by default. It can be changed by assigning Tinit on call
    T = np.copy(t)
    T[0] = Tinit
    for i in range(0,t.shape[0]-1):
        h = dt*dTdt(t[i],T[i])
        T[i+1] = T[i] + h
    return T

def heun(dt,t,Tinit=2): #Tinit is 2 by default. It can be changed by assigning Tinit on call
    T = np.copy(t)
    T[0] = Tinit
    for i in range(0,t.shape[0]-1):
        h1 = dt*dTdt(t[i],T[i])
        h2 = dt*dTdt(t[i]+dt,T[i]+h1)
        T[i+1] = T[i]+0.5*(h1+h2)
    return T

def exct(dt,t,Tinit=2): #Tinit is 2 by default. It can be changed by assigning Tinit on call
    return Tinit*np.exp(-5*t)
    
def dTdt(t,T):
    return -5*T

##Main Program
dt = 0.1
t = np.arange(0.,1.0+dt,dt)

T = exct(dt,t)
plt.plot(t,T,'kx',label= 'Exact') # Exact in black x

T = eul(dt,t)
plt.plot(t,T,'r-',label= 'Euler') # Euler is red line
T = heun(dt,t)
plt.plot(t,T,'g-',label= 'Heun') # Heun is green line
T = rk4(dt,t)
plt.plot(t,T, 'b-',label= 'Runge Kutta') #RK4 is blue line
plt.title('Erdenebeleg Unubold 19B60044')
plt.xlabel('Elapsed time t')
plt.ylabel('Value of the function T')
plt.grid(1)
plt.legend()
plt.show()