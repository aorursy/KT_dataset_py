import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sn

from scipy.integrate import odeint, solve_ivp

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename)) 
# Model Params

###################################################################################################

###################################################################################################

a = 2  #fear factor for country one       

b = 2  #fear factor for country two 



c = 1  #restraint factor for c1

d = 1  #restraint factorrestraint factor for c2



e1 = -5 #strenous factors(?)these are basically here to create an equilibria, as far as i can tell, so cool.

e2 = -5



# Initial values

weaponsC1_0 = +10   # initial weapons held by C1 

weaponsC2_0 = +10   # initial weapons held by C2

###################################################################################################

###################################################################################################





# Bundle parameters for ODE solver

params = [a, b, c, d, e1, e2]



# Bundle initial conditions for ODE solver

y0 = [weaponsC1_0, weaponsC2_0]





# Make time array

tStop = 6

tInc = 0.1

t = np.arange(0., tStop, tInc)

len(t)



def g(t_0):

    return(((t > t_0)*(t < t_0+0.25)))



plt.plot(t,200*(g(0.2)) + 400*g(4))
import numpy as np

import matplotlib.pyplot as plt

import math

from scipy.integrate import odeint









def f(y, t, params):

    weaponsC1, weaponsC2 = y

    a, b, c, d, e1, e2 = params  

    t_0 = 10

    t_1 = 11

    derivs = [ (a*weaponsC2- c*weaponsC1 + e1) + (2*(t > t_0)*(t < t_0+0.5)) + (3*(t > t_1)*(t < t_1+0.5)),       

                (b*weaponsC1 - d*weaponsC2 + e2)] #+ (2*(t > t_0)*(t < t_0+0.5))+ (3*(t > t_1)*(t < t_1+0.5))]

    return derivs













#solve

psoln = odeint(f, y0, t, args=(params,),rtol=1e-10, hmax = 0.01, tcrit = 5)

# Plot results

fig = plt.figure(1, figsize=(8,8))



# Plot C2 Weapons as a function of time

ax1 = fig.add_subplot(311)

ax1.plot(t, psoln[:,0])

ax1.set_xlabel('time')

ax1.set_ylabel('C2 Weapons')

ax1.set_ylim(-1, max(psoln[:,0])+ math.sqrt(max(psoln[:,0])) )





# Plot C1 Weapons as a function of time

ax2 = fig.add_subplot(312)

ax2.plot(t, psoln[:,1])

ax2.set_xlabel('time')

ax2.set_ylabel('C1 Weapons')

ax2.set_ylim(-1, max(psoln[:,0])+ math.sqrt(max(psoln[:,0])) )





# Plot phase-space: C1 Weapons vs C2 Weapons

ax3 = fig.add_subplot(313)

ax3.plot(psoln[:,0], psoln[:,1], '.', ms=1)

ax3.set_xlabel('C2 Weapons')

ax3.set_ylabel('C1 Weapons')



plt.tight_layout()

plt.show()



from pylab import *



x, y = meshgrid(arange(-2, 20, .1), arange(-2, 20, 1))



xdot = a*y-c*x+e1

ydot = b*x-d*y+e2







plt.figure()

plt.streamplot(x, y, xdot, ydot)



x=linspace(-2,20,110)

y=linspace(-2,20,110)
# Model Params

###################################################################################################

###################################################################################################

a = 3  #fear factor for country one       

b = 2  #fear factor for country two 



c = 0.9  #restraint factor for c1

d = 0.78  #restraint factorrestraint factor for c2



e1 = 6 #strenous factors(?)these are basically here to create an equilibria, as far as i can tell, so cool.

e2 = 0.2



# Initial values

weaponsC1_0 = 100    # initial weapons held by C1 

weaponsC2_0 = 2  # initial weapons held by C2

###################################################################################################

###################################################################################################





# Bundle parameters for ODE solver

params = [a, b, c, d, e1, e2]



# Bundle initial conditions for ODE solver

y0 = [weaponsC1_0, weaponsC2_0]





# Make time array

tStop = 3

tInc = 0.1

t = np.arange(0., tStop, tInc)

len(t)



def g(t_0):

    return(((t > t_0)*(t < t_0+0.25)))



k = 200*(g(0.2)) + (400*g(2))
import numpy as np

import matplotlib.pyplot as plt

import math

from scipy.integrate import odeint









def f(y, t, params):

    weaponsC1, weaponsC2 = y

    a, b, c, d, e1, e2 = params  

    t_0 = 0.2

    t_1 = 0.3

    t_2 = 2

    t_3 = 4.5

    t_4 = 6

    (t_5,t_6,t_7,t_8) = (1,1.3,2,2.3)

    derivs = [((1-(weaponsC1/1200)) * (a*weaponsC2- c*weaponsC1 + e1)) + (500*(t > t_0)*(t < t_0+0.25)) + (700*(t > t_2)*(t < t_2+0.5)),# + (200*(t > t_5)*(t < t_5+0.5)) + (200*(t > t_7)*(t < t_7+0.5)),       

              ((1-(weaponsC2/750)) * (b*weaponsC1 - d*weaponsC2 + e2)) + (100*(t > t_0)*(t < t_0+0.5)) + (300*(t > t_2)*(t < t_2+0.5))]# + (100*(t > t_4)*(t < t_4+0.5))]#+ (100*(t > t_6)*(t < t_2+0.6))+ (100*(t > t_8)*(t < t_8+0.5))]

    return derivs













#solve

psoln = odeint(f, y0, t, args=(params,), hmax = 0.01, tcrit = 5)

# Plot results

fig = plt.figure(1, figsize=(8,8))



# Plot C2 Weapons as a function of time

ax1 = fig.add_subplot(311)

ax1.plot(t, psoln[:,0])

ax1.set_xlabel('time')

ax1.set_ylabel('C2 Weapons')

ax1.set_ylim(-1, max(psoln[:,0])+ math.sqrt(max(psoln[:,0])) )





# Plot C1 Weapons as a function of time

ax2 = fig.add_subplot(312)

ax2.plot(t, psoln[:,1])

ax2.plot(t, psoln[:,0])

ax2.plot(t, k)

ax2.set_xlabel('time')

ax2.set_ylabel('C1 Weapons')

ax2.set_ylim(-1, max(psoln[:,0])+ math.sqrt(max(psoln[:,0])) )





#Plot phase-space: C1 Weapons vs C2 Weapons

ax3 = fig.add_subplot(313)

ax3.plot(psoln[:,0], psoln[:,1], '.', ms=1)

ax3.set_xlabel('C2 Weapons')

ax3.set_ylabel('C1 Weapons')



#ax3 = fig.add_subplot(313)

#ax3.plot(t, psoln[:,0])

#ax3.plot(t, psoln[:,1])







plt.tight_layout()

plt.show()
##

def model(y,t):

    k = 0.3

    t_0 = 3

    dydt =  -y + (2*(t > t_0)*(t < t_0+0.4))

    return dydt





# initial condition

y0 = 1



# time points

# Make time array

tStop = 20

tInc = 0.1

t = np.arange(0., tStop, tInc)

len(t)

sol = odeint(model,y0,t)



# plot results

plt.plot(t,sol)

plt.xlabel('time')

plt.ylabel('y(t)')

plt.show()

from pylab import *



x, y = meshgrid(arange(-2, 20, .1), arange(-2, 20, 1))



xdot = 3*y-2*x-5

ydot = 2*x-y-2







plt.figure()

plt.streamplot(x, y, xdot, ydot)



x=linspace(-2,20,110)

y=linspace(-2,20,110)





x, y = meshgrid(arange(-2, 10, .1), arange(-2, 10, 1))



xdot = (1-(x/7)) * (3*y-2*x-5)

ydot = (1-(y/9)) * (2*x-y-2)







plt.figure()

plt.streamplot(x, y, xdot, ydot)



x=linspace(-2,7.5,110)

y=linspace(-2,7.5,110)









show()