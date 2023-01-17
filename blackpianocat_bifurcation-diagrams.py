import numpy as np

import matplotlib.pyplot as plt

import pylab



from pylab import *

from mpl_toolkits import mplot3d
t1 = np.arange(-5, 5, 0.2)

t2 = np.arange(0, 5, 0.2)



plt.plot(t1, 0*t1, 'r-', t2, t2**(1/float(4)), 'b--', t2, -t2**(1/float(4)), 'b-')

plt.xlabel('$\lambda$')

plt.ylabel('x')

plt.show()
a=2



t = np.arange(-1, 1, 0.01)



plt.plot(t, 4*a*t+np.sqrt(4*t**2*(4*a**2+1)), 'b-',t, 4*a*t-np.sqrt(4*t**2*(4*a**2+1)), 'b--')

plt.xlabel('$\lambda$')

plt.ylabel('x')

plt.show()
t1 = np.arange(-2, -1, 0.01)

t2 = np.arange(-1, 0, 0.01)

t3 = np.arange(0, 1, 0.01)



plt.plot(t1, 0*t1, 'b-')

plt.plot(t2, 0*t2, 'r-',t2, np.sqrt(-t2/(t2+1)), 'b--',t2, -np.sqrt(-t2/(t2+1)), 'b-')

plt.plot(t3, 0*t3, 'b--')

plt.xlabel('$\lambda$')

plt.ylabel('x')

plt.show()
t = np.arange(0.923, 5, 0.01)



plt.plot(0.5*(-t+np.sqrt(4*t**3+t**2-4)), t, 'b-',0.5*(-t-np.sqrt(4*t**3+t**2-4)), t, 'b-')

plt.xlabel('$\lambda$')

plt.ylabel('x')

plt.show()
x=np.arange(-4,4,0.1)

l=np.arange(-4,4,1)



for i in l:

    plt.plot(x,1+i**2+i*x,'r-',x,x**3,'b-')

    

plt.xlabel('x')

plt.ylabel('y')

plt.legend(['$f_{1}(x)$','$f_{2}(x)$'])
def f1(mu):

    return np.sqrt(-mu-np.sqrt(mu**2-4*(1-mu)))/np.sqrt(2)

def f2(mu):

    return np.sqrt(-mu+np.sqrt(mu**2-4*(1-mu)))/np.sqrt(2)

def f3(mu):

    return -np.sqrt(-mu-np.sqrt(mu**2-4*(1-mu)))/np.sqrt(2)

def f4(mu):

    return -np.sqrt(-mu+np.sqrt(mu**2-4*(1-mu)))/np.sqrt(2)



t1 = np.arange(-10, 0, 0.01)

t2= np.arange(0, 10, 0.01)



plt.plot(t1,f1(t1),'b-',t1,f2(t1),'b--',t1,f3(t1),'b--',t1,f4(t1),'-b')

plt.plot(t2,f1(t2),'b--',t2,f2(t2),'b-',t2,f3(t2),'b-',t2,f4(t2),'b--')

plt.xlabel('$\mu$')

plt.ylabel('x')

plt.show()
fig = plt.figure();

ax = plt.axes(projection="3d");

def l_function(x, m):

    return m*x-x**4;



x = np.linspace(-2, 2, 30);

m = np.linspace(-2, 2, 30);



X, M = np.meshgrid(x, m);

L = l_function(X, M);



fig = plt.figure();

ax = plt.axes(projection="3d");

ax.plot_wireframe(X, M, L, color='green');

ax.set_xlabel('x');

ax.set_ylabel('$\mu$');

ax.set_zlabel('λ');



plt.show()
mu=np.arange(-5,5,0.01)



plt.plot(mu,3*mu**(4/3)/4**(1/3),'b-',-mu,3*mu**(4/3)/4**(1/3),'b-')

plt.xlabel('$\mu$')

plt.ylabel('$\lambda$')
def f1(l):

    return (-2*l**2+np.sqrt(l**4+2*l**2-4*l+1))/(2*l)

def f2(l):

    return (-2*l**2-np.sqrt(l**4+2*l**2-4*l+1))/(2*l)



t = np.arange(-0.4, 2, 0.1)



plt.plot(t,f1(t),'b-',t,f2(t),'b--')

plt.xlabel('$\lambda$')

plt.ylabel('x')

plt.show()
k=5

l=5



xvalues, yvalues = meshgrid(arange(-6, 6, 0.1), arange(-6, 6, 0.1))

xdot = -xvalues**3-xvalues*yvalues**3+xvalues*yvalues+k

ydot = -yvalues+yvalues*xvalues**2-xvalues**2+l

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
k=5

l=-5



xvalues, yvalues = meshgrid(arange(-6, 6, 0.1), arange(-6, 6, 0.1))

xdot = -xvalues**3-xvalues*yvalues**3+xvalues*yvalues+k

ydot = -yvalues+yvalues*xvalues**2-xvalues**2+l

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
k=-5

l=5



xvalues, yvalues = meshgrid(arange(-6, 6, 0.1), arange(-6, 6, 0.1))

xdot = -xvalues**3-xvalues*yvalues**3+xvalues*yvalues+k

ydot = -yvalues+yvalues*xvalues**2-xvalues**2+l

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()
k=-5

l=-5



xvalues, yvalues = meshgrid(arange(-6, 6, 0.1), arange(-6, 6, 0.1))

xdot = -xvalues**3-xvalues*yvalues**3+xvalues*yvalues+k

ydot = -yvalues+yvalues*xvalues**2-xvalues**2+l

streamplot(xvalues, yvalues, xdot, ydot)

xlabel('x'); ylabel('y')

grid(); show()