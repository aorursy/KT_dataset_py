import numpy as np

import matplotlib.pyplot as plt



def rk(dt,beta,w,wp,Fp,xinit,vinit,tend):

    t = np.arange(0.,tend+dt,dt)

    X = np.copy(t)

    V = np.copy(t)

    X[0] = xinit

    V[0] = vinit

    for i in range(np.shape(t)[0]-1):

        h1 = dt*dxdt(V[i])

        k1 = dt*dvdt(t[i],X[i])

        h2 = dt*dxdt(V[i]+k1/2.)

        k2 = dt*dvdt(t[i]+dt/2.,X[i]+h1/2.)

        h3 = dt*dxdt(V[i]+k2/2.)

        k3 = dt*dvdt(t[i]+dt/2.,X[i]+h2/2.)

        h4 = dt*dxdt(V[i]+k3)

        k4 = dt*dvdt(t[i]+dt,X[i]+h3)

        X[i+1] = X[i]+(h1+2.*h2+2.*h3+h4)/6.

        V[i+1] = V[i]+(k1+2.*k2+2.*k3+k4)/6.

    return X,V,t

    

def dxdt(V):

    return V



def dvdt(t,X):

    return Fp*np.sin(w*t)-wp**2*(1+beta*X**2)*X



dt = 0.001

tend = 100.0

vinit = 0.

xinit = 0.

beta = 0.

wp = np.pi

Fp = 100.0

w = 2.0*np.pi



X,V,t = rk(dt,beta,w,wp,Fp,xinit,vinit,tend)



fig = plt.figure(figsize=(18,6))

plt.plot(t,X)

plt.xlim([0,20])

plt.xlabel('Time (s)')

plt.ylabel('Displacement (m)')

plt.show()