import numpy as np

import matplotlib.pyplot as plt

##LEELAWORASET Woratat

##19B60067



def rk4(dt,beta,w,wp,Fp,xinit,vinit,tend):

    t = np.arange(0.,tend+dt,dt)

    X = np.zeros_like(t)

    V = np.zeros_like(t)

    X[0] = xinit

    V[0] = vinit

    for i in range(0,t.shape[0]-1):

        h1 = dt*dXdt(t[i],X[i],V[i],wp,w,Fp,beta)

        k1 = dt*dVdt(t[i],X[i],V[i],wp,w,Fp,beta)

        h2 = dt*dXdt(t[i]+dt/2.,X[i]+h1/2.,V[i]+k1/2.,wp,w,Fp,beta)

        k2 = dt*dVdt(t[i]+dt/2.,X[i]+h1/2.,V[i]+k1/2.,wp,w,Fp,beta)

        h3 = dt*dXdt(t[i]+dt/2.,X[i]+h2/2.,V[i]+k2/2.,wp,w,Fp,beta)

        k3 = dt*dVdt(t[i]+dt/2.,X[i]+h2/2.,V[i]+k2/2.,wp,w,Fp,beta)

        h4 = dt*dXdt(t[i]+dt,X[i]+h3,V[i]+k3,wp,w,Fp,beta)

        k4 = dt*dVdt(t[i]+dt,X[i]+h3,V[i]+k3,wp,w,Fp,beta)

        X[i+1] = X[i]+(h1+2.*h2+2.*h3+h4)/6.

        V[i+1] = V[i]+(k1+2.*k2+2.*k3+k4)/6.

    return X,V,t

    

def dXdt(t,X,V,wp,w,Fp,beta):

    return V



def dVdt(t,X,V,wp,w,Fp,beta):

    return Fp*np.sin(w*t)-((wp**2)*(1+(beta*(X**2)))*X)



dt = 0.001

tend = 100.0

vinit = 0.

xinit = 0.

beta = 0.

wp = np.pi

Fp = 100.0

w = 2.0*np.pi



X,V,t = rk4(dt,beta,w,wp,Fp,xinit,vinit,tend)



fig = plt.figure(figsize=(18,6))

plt.plot(t,X)

plt.xlim([0,20])

plt.xlabel('Time (s)')

plt.ylabel('Displacement (m)')

plt.show()