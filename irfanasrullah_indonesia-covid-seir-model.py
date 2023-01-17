import numpy as np # linear algebra

import pandas as pd # data processing

from scipy.integrate import odeint

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline 

!pip install mpld3

import mpld3

mpld3.enable_notebook()
def plotseird(t, S, E, I, R, D=None, L=None, R0=None, Alpha=None):

  f, ax = plt.subplots(1,1,figsize=(10,4))

  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')

  ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')

  ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')

  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

  if D is not None:

    ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')

    ax.plot(t, S+E+I+R+D, 'c--', alpha=0.7, linewidth=2, label='Total')

  else:

    ax.plot(t, S+E+I+R, 'c--', alpha=0.7, linewidth=2, label='Total')



  ax.set_xlabel('Time (days)')



  ax.yaxis.set_tick_params(length=0)

  ax.xaxis.set_tick_params(length=0)

  ax.grid(b=True, which='major', c='w', lw=2, ls='-')

  legend = ax.legend(borderpad=2.0)

  legend.get_frame().set_alpha(0.5)

  for spine in ('top', 'right', 'bottom', 'left'):

      ax.spines[spine].set_visible(False)

  if L is not None:

      plt.title("PSBB (Large-Scale Social Distancing) after {} days".format(L))

  plt.show();



  #if R0 is not None or CFR is not None:

  #  f = plt.figure(figsize=(12,4))

  

  if R0 is not None:

    # sp1

    ax1 = f.add_subplot(121)

    ax1.plot(t, R0, 'b--', alpha=0.7, linewidth=2, label='R_0')



    ax1.set_xlabel('Time (days)')

    ax1.title.set_text('R_0 over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax1.yaxis.set_tick_params(length=0)

    ax1.xaxis.set_tick_params(length=0)

    ax1.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax1.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

      ax.spines[spine].set_visible(False)

    

  if Alpha is not None:

    # sp2

    ax2 = f.add_subplot(122)

    ax2.plot(t, Alpha, 'r--', alpha=0.7, linewidth=2, label='alpha')



    ax2.set_xlabel('Time (days)')

    ax2.title.set_text('fatality rate over time')

    # ax.set_ylabel('Number (1000s)')

    # ax.set_ylim(0,1.2)

    ax2.yaxis.set_tick_params(length=0)

    ax2.xaxis.set_tick_params(length=0)

    ax2.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax2.legend()

    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):

      ax.spines[spine].set_visible(False)



    plt.show();
def deriv(y, t, N, beta, gamma, delta):

    S, E, I, R = y

    dSdt = -beta * S * I / N

    dEdt = beta * S * I / N - delta * E

    dIdt = delta * E - gamma * I

    dRdt = gamma * I

    return dSdt, dEdt, dIdt, dRdt
N = 250000000

D = 5.0 # infections lasts five days

gamma = 1.0 / D

delta = 1.0 / 5.0  # incubation period of five days

R_0 = 3.86 # angka reproduksi rata-rata dengan social distancing (R0) based on Pei Jun Zhao

beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma

#beta = 1000000 # expected amount of people an infected person infects per day based on Pei Jun Zhao is 1 million

S0, E0, I0, R0 = N-1, 1, 0, 0  # initial conditions: one exposed
t = np.linspace(0, 300, 300) # Grid of time points (in days)

y0 = S0, E0, I0, R0 # Initial conditions vector



# Integrate the SIR equations over the time grid, t.

ret_exposed = odeint(deriv, y0, t, args=(N, beta, gamma, delta))

S, E, I, R = ret_exposed.T
df_exposed = pd.DataFrame(ret_exposed)

df_exposed.columns = ['Susceptible','Exposed','Infected','Recovered']

print(df_exposed[100:110])
# Get a series containing maximum value of each column

max_df_exposed = df_exposed.max()

df_exposed.idxmax(axis=0, skipna=True)



#print('Maximum value in each column : ')

#print(max_df_death)



maxValue_df_exposed = df_exposed.idxmax()

 

print("Max values of columns are at row index position :")

print(maxValue_df_exposed)
plotseird(t, S, E, I, R)
def deriv(y, t, N, beta, gamma, delta, alpha, rho):

    S, E, I, R, D = y

    dSdt = -beta * S * I / N

    dEdt = beta * S * I / N - delta * E

    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I

    dRdt = (1 - alpha) * gamma * I

    dDdt = alpha * rho * I

    return dSdt, dEdt, dIdt, dRdt, dDdt
N = 250_000_000

D = 5.0 # infections lasts five days

gamma = 1.0 / D

delta = 1.0 / 5.0  # incubation period of five days

R_0 = 3.86 # Covid R0 based on Pei Jun Zhao

beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma

alpha = 0.08  # 8% death rate in Indonesia

rho = 1/12  # 9 days from infection until death

S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed
t = np.linspace(0, 300, 300) # Grid of time points (in days)

y0 = S0, E0, I0, R0, D0 # Initial conditions vector



# Integrate the SIR equations over the time grid, t.

ret_death = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))

S, E, I, R, D = ret_death.T
df_death = pd.DataFrame(ret_death)

df_death.columns = ['Susceptible','Exposed','Infected','Recovered', 'Death']

print(df_death[291:301])
# Get a series containing maximum value of each column

max_df_death = df_death.max()

df_death.idxmax(axis=0, skipna=True)



#print('Maximum value in each column : ')

#print(max_df_death)



maxValue_df_death = df_death.idxmax()

 

print("Max values of columns are at row index position :")

print(maxValue_df_death)
plotseird(t, S, E, I, R, D)
def deriv(y, t, N, beta, gamma, delta, alpha, rho):

    S, E, I, R, D = y

    dSdt = -beta(t) * S * I / N

    dEdt = beta(t) * S * I / N - delta * E

    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I

    dRdt = (1 - alpha) * gamma * I

    dDdt = alpha * rho * I

    return dSdt, dEdt, dIdt, dRdt, dDdt
L = 40 # lockdown or social distancing applied after L days

N = 250000000 # population

D = 5.0 # infections first five days

gamma = 1.0 / D

delta = 1.0 / 5.0  # incubation period of five days

def R_0(t):

    return 3.86 if t < L else 1.9 

def beta(t):

    return R_0(t) * gamma



alpha = 0.08  # 8% covid death rate in Indonesia

rho = 1/9  # 9 days from infection until death

S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed
t = np.linspace(0, 300, 300) # Grid of time points (in days)

y0 = S0, E0, I0, R0, D0 # Initial conditions vector



# Integrate the SIR equations over the time grid, t.

ret_psbb = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))

S, E, I, R, D = ret_psbb.T
df_psbb = pd.DataFrame(ret_psbb)

df_psbb.columns = ['Susceptible','Exposed','Infected','Recovered', 'Death']

print(df_psbb[275:300])
plotseird(t, S, E, I, R, D, L)
def deriv(y, t, N, beta, gamma, delta, alpha_opt, rho):

    S, E, I, R, D = y

    def alpha(t):

        return s * I/N + alpha_opt



    dSdt = -beta(t) * S * I / N

    dEdt = beta(t) * S * I / N - delta * E

    dIdt = delta * E - (1 - alpha(t)) * gamma * I - alpha(t) * rho * I

    dRdt = (1 - alpha(t)) * gamma * I

    dDdt = alpha(t) * rho * I

    return dSdt, dEdt, dIdt, dRdt, dDdt
N = 250_000_000

D = 5.0 # infections lasts four days

gamma = 1.0 / D

delta = 1.0 / 5.0  # incubation period of five days



R_0_start, k, x0, R_0_end = 3.86, 0.5, 50, 1.99



def logistic_R_0(t):

    return (R_0_start-R_0_end) / (1 + np.exp(-k*(-t+x0))) + R_0_end



def beta(t):

    return logistic_R_0(t) * gamma



alpha_by_agegroup = {"0-9": 0.005, "10-29": 0.030, "30-49": 0.187, "50-69": 0.600, "70+": 0.177}

proportion_of_agegroup = {"0-9": 0.017, "10-29": 0.189, "30-49": 0.392, "50-69": 0.346, "70+": 0.055}

s = 0.01

alpha_opt = sum(alpha_by_agegroup[i] * proportion_of_agegroup[i] for i in list(alpha_by_agegroup.keys()))



rho = 1/9  # 9 days from infection until death

S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed
t = np.linspace(0, 300, 300) # Grid of time points (in days)

y0 = S0, E0, I0, R0, D0 # Initial conditions vector



# Integrate the SIR equations over the time grid, t.

ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha_opt, rho))

S, E, I, R, D = ret.T

R0_over_time = [logistic_R_0(i) for i in range(len(t))]  # to plot R_0 over time: get function values

Alpha_over_time = [s * I[i]/N + alpha_opt for i in range(len(t))]  # to plot alpha over time
plotseird(t, S, E, I, R, D, R0=R0_over_time, Alpha=Alpha_over_time)