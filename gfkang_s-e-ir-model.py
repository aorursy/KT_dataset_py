# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 1223000
# Initial number of exposed, infected, recovered individuals, E0, I0 and R0.
E0, I0, R0, Fatal0, Hosp0 = 0,1,0,0,0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0 -E0
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#Rt is measure of contagiousness: the number of secondary infections each infected individual produces.
#Tinc is length of incubation period
#Tinf is duration patient is infectious
#beta is Rt/Tinf
#sigma is 1/Tinc
#gamma is 1/Tinf
sigma, beta, gamma = 1./5, 0.55, 1./14 
D_death = 25

#death rate
death_rate = 0.015
#case fatality rate
p_fatal = 0.02
#estimated hospitalization rate
hosp_rate = 0.03
# A grid of time points (in days)
t = np.linspace(0, 40, 40)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, sigma):
    S, E, I, R, Fatal, Hosp = y
    dSdt = -beta * S * I / N
    dEdt = (-sigma * E) + (beta * S * I / N)
    dIdt = (sigma * E) - (gamma * I)
    dRdt = gamma * I
    dFatal = p_fatal*gamma*I  - (1/D_death)*Fatal0
    dHosp = hosp_rate*I
    return dSdt, dEdt, dIdt, dRdt, dFatal, dHosp

# Initial conditions vector
y0 = S0, E0, I0, R0, Fatal0, Hosp0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, sigma))
#print(ret)

new_ret = pd.DataFrame({'Susceptible': ret[:, 0], 'Exposed': ret[:, 1],'Infected': ret[:, 2], 'Recovered': ret[:, 3], 'Fatal':
                       ret[:, 4], 'Hospitalized': ret[:, 5]})
S, E, I, R, Fatal, Hosp = ret.T
print(new_ret[0:30])


axis_bgcolor='#dddddd'

# Plot the data on three separate curves for S(t), E(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, axisbelow=True)
#ax.plot(t, S/1000000, 'b', alpha=0.5, lw=2, label='Susceptible')
#ax.plot(t, E/1000000, 'y', alpha=0.5, lw=2, label='Exposed')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, Hosp, 'g', alpha=0.5, lw=2, label='Hospitalized')
ax.plot(t, Fatal, 'b', alpha=0.5, lw=2, label='Deaths')
#ax.plot(t, R/1000000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.set_ylabel('Number of Cases')
ax.set_ylim(0,300)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
import csv

#https://www.kaggle.com/sandraho1216/coviddailydata
real_data = pd.read_csv('../input/new-infections/New Infections Death Hospitalized - Sheet1.csv')  

#for row in real_data:
#    print(row)
    #print(col)
    #print(col['Hospitalized'])
real_data = real_data.drop(real_data.index[[0,1,2,3]])
real_data = real_data.drop('Unnamed: 0', axis=1)
real_data.columns = ['Date', 'Infections','Deaths','Hospitalized']
real_data.dropna()
real_data
    
#ax.plot(t, real_data, 'b', alpha=0.5, lw=2, label='Real Data')
#real_data.plot(kind='line',x='Date',y='Infections',ax=ax)