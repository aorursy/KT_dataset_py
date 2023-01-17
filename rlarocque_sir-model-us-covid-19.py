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
# From SciPython book: https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/



# Plugging in estimates for Novel Coronavirus and US.



import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt



# The SIR model differential equations.

def deriv(y, t, N, beta, gamma):

    S, I, R = y

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def plot_sir_curves(title, N, I0, R0, beta, gamma):

    R_0 = beta / gamma



    # A grid of time points (in days)

    t = np.linspace(0, 365 * 2, 1000)



    # Initial conditions vector

    y0 = S0, I0, R0

    # Integrate the SIR equations over the time grid, t.

    ret = odeint(deriv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T



    # Plot the data on three separate curves for S(t), I(t) and R(t)

    fig = plt.figure(facecolor='w', figsize=(20, 10))



    ax = fig.add_subplot(111) #, axisbelow=True)

    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')

    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')

    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')

    ax.set_xlabel('Time /days')

    ax.set_ylabel('Number')

    ax.set_ylim(0, N * 1.1)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))



    legend = ax.legend()

    legend.get_frame().set_alpha(1.0)



    ax.minorticks_on()

    ax.grid(b=True, axis='y', which='minor', linestyle=':')

    ax.grid(b=True, axis='y', which='major', lw=2, ls='-')

    ax.grid(b=True, axis='x', which='major', lw=2, ls='-', alpha=0.5)



    plt.title("{:}\n(R_0: {:.3f}, beta: {:.3f})".format(title, R_0, beta))



    plt.show()
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 4

beta = R_0 * gamma



plot_sir_curves("US - Worst Case Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 1.4

beta = R_0 * gamma



plot_sir_curves("US - Optimistic Estimate, No Lockdown Scenario", N, I0, R0, beta, gamma)
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 2.5

beta = R_0 * gamma



plot_sir_curves("US - Reasonable $R_0$ with no lockdown", N, I0, R0, beta, gamma)
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 2.5

beta = R_0 * gamma * (1 - 0.25)



plot_sir_curves("US - Lockdown 25% Effective Scenario", N, I0, R0, beta, gamma)
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 2.5

beta = R_0 * gamma * (1 - 0.4)



plot_sir_curves("US - Lockdown 40% Effective Scenario", N, I0, R0, beta, gamma)
# Total population, N.

N = 327000000



# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 3500, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0



# mean recovery rate, gamma, (in 1/days).

gamma = 1./14



# Contact rate, beta (percent of population touched+infected by an infected person daily).

# Also, R_0 = beta / gamma

R_0 = 2.5

beta = R_0 * gamma * (1 - 0.60)



plot_sir_curves("US - Lockdown 60% Effective Scenario", N, I0, R0, beta, gamma)