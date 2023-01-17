# Código original de: APMonitor

import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import ipywidgets as wg
from IPython.display import display
n = 1000 # time points to plot
tf = 20.0 # final time
SP_start = 2.0 # time of set point change

def process(x,t,u):
    y = x[0]
    dydt = x[1]
    
    Kp = 8.0
    taup = 1.5
    thetap = 1.0
    zeta = 1.5
    
    if t<(thetap+SP_start):
        dydt = 0.0  # time delay
        dydt2 = 0.0
    else:
        dydt2 = (-2.0*zeta*taup*dydt - y + Kp*u)/taup**2
    return [dydt, dydt2]

def pidPlot(Kc = 0.0,tauI = 100.0 ,tauD = 0.0):
    t = np.linspace(0,tf,n) # create time vector
    P= np.zeros(n)          # initialize proportional term
    I = np.zeros(n)         # initialize integral term
    D = np.zeros(n)         # initialize derivative term
    e = np.zeros(n)         # initialize error
    OP = np.zeros(n)        # initialize controller output
    PV = np.zeros(n)        # initialize process variable
    SP = np.zeros(n)        # initialize setpoint
    SP_step = int(SP_start/(tf/(n-1))+1) # setpoint start
    SP[0:SP_step] = 0.0     # define setpoint
    SP[SP_step:n] = 40.0     # step up
    x0 = [0,0]                # initial condition
    # loop through all time steps
    for i in range(1,n):
        # simulate process for one time step
        ts = [t[i-1],t[i]]         # time interval
        x = odeint(process,x0,ts,args=(OP[i-1],))  # compute next step
        x0 = x[1]                  # record new initial condition
        # calculate new OP with PID
        PV[i] = x0[0]               # record PV
        e[i] = SP[i] - PV[i]       # calculate error = SP - PV
        dt = t[i] - t[i-1]         # calculate time step
        P[i] = Kc * e[i]           # calculate proportional term
        I[i] = I[i-1] + (Kc/tauI) * e[i] * dt  # calculate integral term
        D[i] = -Kc * tauD * (PV[i]-PV[i-1])/dt # calculate derivative term
        OP[i] = P[i] + I[i] + D[i] # calculate new controller output
        
    # plot PID response
    plt.figure(1,figsize=(15,7))
    plt.subplot(2,2,1)
    plt.plot(t,SP,'k-',linewidth=2,label='Setpoint (SP)')
    plt.plot(t,PV,'r:',linewidth=2,label='Variável de processo (PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,2)
    plt.plot(t,P,'g.-',linewidth=2,label=r'Proporcional = $K_c \; e(t)$')
    plt.plot(t,I,'b-',linewidth=2,label=r'Integral = $\frac{K_c}{\tau_I} \int_{i=0}^{n_t} e(t) \; dt $')
    plt.plot(t,D,'r--',linewidth=2,label=r'Derivativo = $-K_c \tau_D \frac{d(PV)}{dt}$')    
    plt.legend(loc='best')
    plt.subplot(2,2,3)
    plt.plot(t,e,'m--',linewidth=2,label='Erro (e=SP-PV)')
    plt.legend(loc='best')
    plt.subplot(2,2,4)
    plt.plot(t,OP,'b--',linewidth=2,label='Controller Output (OP)')
    plt.legend(loc='best')
    plt.xlabel('time')
wg.interact(pidPlot, Kc = (0.0, 20.0, 0.1), tauI = (0.0, 100,0.5), tauD = (0.0, 20.0, 0.1))

# pidPlot( Kc = 0.0, tauI = 100.0, tauD = 0.0)