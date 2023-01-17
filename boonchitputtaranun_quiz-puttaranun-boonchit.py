import numpy as np
import matplotlib.pyplot as plt

def rk(dt,beta,w,wp,Fp,xinit,vinit,tend):
    t = np.arange(0.,tend+dt,dt)
    X = np.copy(t)
    V = np.copy(t)
    for i in range(0,t.shape[0]-1):
        h1 = dt * dXdt(t[i],X[i],V[i],w,wp,Fp,beta)
        k1 = dt * dVdt(t[i],X[i],V[i],w,wp,Fp,beta)
        h2 = dt * dXdt(t[i]+dt/2.,X[i]+h1/2,V[i]+k1/2,w,wp,Fp,beta)
        k2 = dt * dVdt(t[i]+dt/2.,X[i]+h1/2,V[i]+k1/2,w,wp,Fp,beta)
        h3 = dt * dXdt(t[i]+dt/2.,X[i]+h2/2,V[i]+k2/2,w,wp,Fp,beta)
        k3 = dt * dVdt(t[i]+dt/2.,X[i]+h2/2,V[i]+k2/2,w,wp,Fp,beta)
        h4 = dt * dXdt(t[i]+dt,X[i]+h3,V[i]+k3,w,wp,Fp,beta)
        k4 = dt * dVdt(t[i]+dt,X[i]+h3,V[i]+k3,w,wp,Fp,beta)
        X[i+1] = X[i]+(h1+2.*h2+2.*h3+h4)/6
        V[i+1] = V[i]+(k1+2.*k2+2.*k3+k4)/6
    return X,V,t
    
def dXdt(t,X,V,w,wp,Fp,beta):
    return V

def dVdt(t,X,V,w,wp,Fp,beta):
    return (Fp*np.sin(w*t))-(wp*wp*(1+(beta*X*X))*X)

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