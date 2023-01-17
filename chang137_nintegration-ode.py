import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import imageio
import numpy as np
#import pylab

def gaussian(x, s, m):
    return 1./(np.sqrt(2.*np.pi)*s) * np.exp(-0.5*((x-m)/s)**2)

m = 0
s = np.linspace(0.5,5,3)
x, dx = np.linspace(-10,10,1000, retstep=True)

x = x[:,np.newaxis]
y = gaussian(x,s,m)

h = 1.e-6
dydx = (gaussian(x+h, s, m) - gaussian(x-h, s, m))/2/h
int_y = np.sum(gaussian(x, s, m), axis=0) * dx
print(int_y)

#pylab.plot(x, y)
#pylab.plot(x, dydx)
#pylab.show()

#plt.plot(x, y)
plt.plot(x, dydx)
plt.show()

import numpy as np
#from scipy.optimize import fsolve
from scipy.optimize import least_squares
parameter=dict()
parameter['u_h']=6.0
parameter['k_oh']=0.20
parameter['k_s']=20.0
parameter['k_h']=3.0
parameter['k_x']=0.03
parameter['Y_h']=0.67
parameter['f_p']=0.08
parameter['b_h']=0.62 

Bulk_DO=2.0 #mg/L

#influent components:
infcomp=[56.53,182.9,16.625] #mgCOD/l

Q=684000 #L/hr
V=1040000 #l



def steady(z,*args):
    Ss=z[0]
    Xs=z[1]
    Xbh=z[2]
    def monod(My_S,My_K):
        return My_S/(My_S+My_K)

    #Conversion rates
    #Conversion of Ss
    r1=((-1/parameter['Y_h'])*parameter['u_h']*monod(Ss,parameter['k_s'])\
        +parameter['k_h']*monod(Xs/Xbh,parameter['k_x'])*monod(Bulk_DO,parameter['k_oh']))\
        *Xbh*monod(Bulk_DO,parameter['k_oh'])

    #Conversion of Xs
    r2=((1-parameter['f_p'])*parameter['b_h']-parameter['k_h']*monod(Xs/Xbh,parameter['k_x']))*Xbh

    #Conversion of Xbh
    r3=(parameter['u_h']*monod(Ss,parameter['k_s'])*monod(Bulk_DO,parameter['k_oh'])-parameter['b_h'])*Xbh

    f=np.zeros(3)
    f[0]=Q*(infcomp[0]-Ss)+r1*V
    f[1]=Q*(infcomp[1]-Xs)+r2*V
    f[2]=Q*(infcomp[2]-Xbh)+r3*V
    return f
initial_guess=(0.1,0.1,0.1)
#soln=fsolve(steady,initial_guess,args=parameter)
soln = least_squares(steady, initial_guess, bounds=[(0,0,0),(np.inf,np.inf,np.inf)], args=parameter)
print (soln)

from scipy.integrate import solve_ivp
def rhs(s, v): 
    return [-12*v[2]**2, 12*v[2]**2, 6*v[0]*v[2] - 6*v[2]*v[1] - 36*v[2]]
res = solve_ivp(rhs, (0, 0.1), [2, 3, 4])
print(res)

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
%matplotlib inline

def rhs(s, v): 
    return [-12*v[2]**2, 12*v[2]**2, 6*v[0]*v[2] - 6*v[2]*v[1] - 36*v[2]]
res = solve_ivp(rhs, (0, 0.1), [2, 3, 4])
print(res)

plt.plot(res.t, res.y.T)
plt.show()

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
import imageio

from scipy import integrate as integrate
from scipy.integrate import odeint
from scipy.integrate import quad

def ab(a,b,phia,phib):
    return 1-np.cos(a)*np.cos(b)-np.cos(phia-phib)*np.sin(a)*np.sin(b)

def AB(a,b,phia,phib):
    return 1+np.cos(a)*np.cos(b)-np.cos(phia-phib)*np.sin(a)*np.sin(b)

def U(a,b,j,phia,phib,phij,L):
    return 2**(L/2)*np.cos(j)**L*np.sqrt(ab(a,b,phia,phib)/AB(a,j,phia,phij)/ab(j,b,phij,phib))**L
# define 1-dressed gluon
def integrand(a,b,j,phia,phib,phij,L):
    return ab(a,b,phia,phib)/ab(a,j,phia,phij)/ab(j,b,phij,phib)*(U(a,b,j,phia,phib,phij,L)-1)*np.sin(j)
print(integrand(0,np.pi,np.pi/2,0,np.pi,np.pi,2))

def int1(a,b,j,phia,phib,L):
    #return quad(integrand, 0, 2*np.pi, args=(a,b,j,phia,phib,L))[0]
    return quad(lambda phij: integrand(a,b,j,phia,phib,phij,L), 0, 2*np.pi)[0]
print(int1(0,np.pi,np.pi/2,0,np.pi,1)/4/np.pi)
def int2(a,b,phia,phib,L):
    return quad(lambda j: int1(a,b,j,phia,phib,L), 10**-5, np.pi/2)[0]
print(int2(0,np.pi,0,np.pi,1)/4/np.pi)


# function that returns dy/dt
def model(y,t):
    a=0.
    b=np.pi
    phia=0.
    phib=np.pi
    dydt = int2(a,b,phia,phib,t)/4/np.pi
    return dydt
#print(model(1,1))

# initial condition
y0 = 1

# time points
t = np.linspace(0,5)

# solve ODE
y = odeint(model,y0,t)

# plot results
euler_constant = 0.57721566490153286060 # Euler Mascheroni Constant

def psy_analytic(x):
    '''
        Profile of the exact solution
    '''
    return 1 - (euler_constant * x + torch.lgamma(1 + x)) / 2.

x0 = torch.unsqueeze(torch.linspace(0, 5, 20), dim=1)  # x data (tensor), shape=(100, 1)
#x = x0.clone().detach(requires_grad=True)
x=x0.clone().detach().requires_grad_(True)
ya = psy_analytic(x)

# view data
plt.figure(figsize=(10,4))
plt.scatter(x.data.numpy(), ya.data.numpy(), color = "orange")
plt.plot(t,y)
plt.xlabel('log')
plt.ylabel('y(t)')
plt.show()