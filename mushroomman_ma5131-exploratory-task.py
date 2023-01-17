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

import pandas as pd

from scipy.integrate import odeint

import matplotlib.pyplot as plt



# Total population, N.

N = 250000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 1, 0



# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



contact_rate = 1.7

# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).

gamma = 0.143 #we will see which model fits past data best

beta = gamma * contact_rate



# A grid of time points (in days)

t = np.linspace(0, 160, 160)



# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



# Initial conditions vector

y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, beta, gamma))

S, I, R = ret.T



# S, I, R are the arrays of the predicted values of susceptible, infected, and recovered (and dead) for each day

# we will compare this with past data from the excel file and see if they match



# Extracting past data

data = pd.read_csv("../input/covid19-singapore-may-2/pastdata.csv")

df = pd.DataFrame(data)

print(df)



# Day 00 is 23rd Jan (first day of infection)

# Day 09 is 1st Feb

# Day 38 is 1st Mar

# Day 69 is 1st Apr

# Day 99 is 1st May

print(data.iloc[97:99])

print(str(I[97]) + " " + str(I[98]))

print(str(R[97]) + " " + str(R[98]))



# Calculating residuals

#Sdiff = data.susceptible - S[:91]

#Idiff = data.infected - I[:91]

#Rdiff = data.recovered - R[:91]



#Sdiff = Sdiff * Sdiff

#Idiff = Idiff * Idiff

#Rdiff = Rdiff * Rdiff



#errorS = Sdiff.sum()

#errorI = Idiff.sum()

#errorR = Rdiff.sum()



#print(str(errorS) + " " + str(errorI) + " " + str(errorR))



# Plot the data on three separate curves for S(t), I(t) and R(t)

fig = plt.figure(facecolor='w')

ax = fig.add_subplot(111, axisbelow=True)

ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='predictedS')

ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='predictedI')

ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='predictedR')



#ax.plot(t, data.susceptible/N, 'm', alpha=0.5, lw=2, label='S')

ax.plot(t, data.infected/N, 'y', alpha=0.5, lw=2, label='I')

ax.plot(t, data.recovered/N, 'k', alpha=0.5, lw=2, label='R')



ax.set_xlabel('Time /days')

ax.set_ylabel('Fraction of Population')

ax.set_ylim(0,0.2)

ax.yaxis.set_tick_params(length=0)

ax.xaxis.set_tick_params(length=0)

ax.grid(b=True, which='major', c='w', lw=2, ls='-')

legend = ax.legend()

legend.get_frame().set_alpha(0.5)

for spine in ('top', 'right', 'bottom', 'left'):

    ax.spines[spine].set_visible(False)

plt.show()



days = ['20-Apr', '21-Apr', '22-Apr', '23-Apr', '24-Apr', '25-Apr', '26-Apr',

        '27-Apr', '28-Apr', '29-Apr', '30-Apr', '01-May', '02-May', '03-May',

        '04-May', '05-May', '06-May', '07-May', '08-May', '09-May', '10-May',

        '11-May', '12-May', '13-May', '14-May', '15-May', '16-May', '17-May']

for count in range(87, 115):

    print(days[count-87] + " " + str(I[count+1] + R[count+1] - I[count] - R[count]))

    