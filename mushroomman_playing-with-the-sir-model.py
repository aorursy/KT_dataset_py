# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt



# We will be using the odeint library from the scipy module to integrate the differential equations



# This is the function which will help us to find the values for the susceptible, infected and recovered

# at a certain time t, by solving the differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



# We shall now set our parameters for the model

# Feel free to change these parameters and see how the model changes

N = 15000



S0 = 14990

I0 = 10

R0 = 0



b = 0.243

k = 0.143



no_of_days = 100



# Calculating Basic Reproduction Number Ro

reproduction_number = b / k



# A grid of time points (in days)

t = np.linspace(0, no_of_days + 1, no_of_days + 1)



# Initial conditions vector

y0 = S0, I0, R0



# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, b, k))

S, I, R = ret.T



# Rounding off

S = np.round(np.array(S))

I = np.round(np.array(I))

R = np.round(np.array(R))



# Plot the data on three separate curves for S(t), I(t) and R(t)

fig = plt.figure(facecolor='w')

ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')

ax.set_xlabel('Time /days')

ax.set_ylabel('Number')

ax.set_ylim(0,1.2*N)

ax.yaxis.set_tick_params(length=0)

ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')

legend = ax.legend()

legend.get_frame().set_alpha(0.5)

for spine in ('top', 'right', 'bottom', 'left'):

    ax.spines[spine].set_visible(False)

plt.show()



# Printing initial conditions

print("Day 0")

print("Initial Cases: " + str(I0))

print("Active Cases: " + str(I0))

print("Total Cases: " +str(I0) + "\n")



for count in range(no_of_days):

    new_infected = int(I[count+1] + R[count+1] - I[count] - R[count])

    new_recovered = int(R[count+1] - R[count])

    active = int(I[count+1])

    total = int(I[count+1] + R[count+1])

    print("Day " + str(count+1))

    print("New Cases: " + str(new_infected))

    print("New Recoveries: " + str(new_recovered))

    print("Active Cases: " + str(active))

    print("Total Cases: " + str(total) + "\n")