import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

from scipy import interpolate,optimize,integrate,signal

import matplotlib.pyplot as plt

def f_tanh(x,Pm,R,L,phi):

    return Pm*(1+R*np.tanh((x-phi)*np.pi/L))





def f_fit_tanh(x,P1,P2,P3,T_insp,C1,C2,x0):

    deltaP =P2-P1

    Pm = (P2+P1)/2

    R = deltaP / Pm/2

    f1 = f_tanh(x,Pm,R,C1,x0)

    deltax = T_insp

    Pm2 = (P2+P3)/2

    deltaP2 = P3-P2

    R2 = deltaP2 / Pm2/2

    f2 = f_tanh(x,Pm2,R2,C2,deltax)

    

    

    fx = np.copy (f1)

    if x<x0+deltax:

        fx = f1

    else:

        fx =f2

    return fx 
fig3,ax3 = plt.subplots()

times = np.linspace(-5,15,200)

P1 = 5

P2 = 10

P3 = 5

T_insp = 4

C1,C2 = [2,1]

t0 = 0

ft = np.zeros_like(times)



for i,ti in enumerate(times):

    ft[i] = f_fit_tanh(ti,P1,P2,P3,T_insp,C1,C2,t0)



ax3.plot(times,ft);