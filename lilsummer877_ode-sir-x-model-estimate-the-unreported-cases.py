%matplotlib inline

import matplotlib.pyplot as plt

# import networkx as nx

# import ndlib.models.epidemics as ep

import os

import pandas as pd

import numpy as np

from scipy.integrate import odeint

from scipy.optimize import fsolve

from scipy.integrate import simps

from numpy import trapz

plt.style.use('ggplot')
## refer to table 1, Hubei

r_0_theory = 6.2

r_0_eff  = 3.28



## T_i is the time between infection and quanrantine

T_i_theory = 8

alpha = r_0_theory / T_i_theory



## 

beta = alpha/r_0_theory

print('alpha:', alpha, 'beta:', beta)

# Q and P are from table 1

Q= 0.47

P = 0.66

def get_kapp(k):

    kappa_0 = k[0]

    kappa = k[1]

    

    F = np.empty((2))

    F[0] = (kappa_0+ kappa)/(beta+kappa_0 + kappa) - Q

    F[1] = kappa_0/(kappa+kappa_0) -P

    

    return F





kappaGuess = np.array([1,1])

(kappa_0, kappa) = fsolve(get_kapp,kappaGuess)

print('kappa_0: ',kappa_0, 'kappa: ',kappa)
# solve the system dy/dt = f(y, t)

def f(y, t):

    S = y[0]

    I = y[1]

    X = y[2]

     # the model equations (see Munz et al. 2009)

    f0 = -alpha* S* I - kappa_0* S

    f1 = alpha* S* I - beta* I  - kappa_0* I - kappa*I 

    f2 = (kappa + kappa_0)* I 

    return [f0, f1, f2]
## import data

train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

train['Date'] = pd.to_datetime(train.Date)
train[(train['Country/Region'] == 'China')&(train['Province/State']=='Hubei')].head(10)
hubei_df = train[(train['Country/Region'] == 'China')&(train['Province/State']=='Hubei')]
dti = pd.date_range('2020-01-21', '2020-03-10', freq='D')

len(dti)
## according to the paper page 6

# initial conditions

# use the number of cases 444

X0 = 444/(57.1*10**6)

I0 = 2.55*X0

S0 = 1-I0-X0

# initial death population

y0 = [S0, I0, X0]     # initial condition vector

t  = np.linspace(0, 50., 50)  # time grid

#print(y0)

# solve the DEs

soln = odeint(f, y0, t)

S = soln[:, 0]

I = soln[:, 1]

X = soln[:, 2]
# plot results

plt.figure(figsize=(15, 10))

plt.title('Hubei Confirmed Cases vs Model')

#plt.plot(t, S, label='Living')

plt.plot(dti, I*57.1*10**6/(4.44/2.55), label='Infected (I)')

plt.plot(dti, X*57.1*10**6/(4.44/2.55), label='Quanrantined (X)')

plt.plot(hubei_df.Date, hubei_df.ConfirmedCases, 'o')

plt.vlines(x=pd.datetime(2020,2,7), ymin=0, ymax=70000, linestyles='dashed', color='grey')

plt.text(x=pd.datetime(2020,2,2), text='02/07/2020', y=30000,s=10)

plt.xlabel('Days from outbreak')

plt.ylabel('Population')

plt.legend(loc=0)

plt.xticks(rotation=90)

plt.show()
len(pd.date_range('2020-01-21', '2020-02-07'))
total_number_infection = (I*57.1*10**6/(4.44/2.55))[:18]
total_number_confirm = hubei_df.ConfirmedCases[:18]
under_report = total_number_infection - total_number_confirm
under_report_pcnt = under_report/total_number_infection*100
under_report.sum()/total_number_infection.sum()
plt.figure(figsize=(10,5))

plt.title('Confirmed and Modeled Infection')

plt.bar(hubei_df.Date[:18], total_number_infection, alpha=.8, color='grey', label='model_infection')

plt.bar(hubei_df.Date[:18], total_number_confirm, alpha=.5, color='red', label='confirmed')

plt.xticks(rotation=90)

plt.legend()

plt.show()
plt.figure(figsize=(10,5))

plt.title('% of under-reported cases with respect to the toatl number of modeled infection cases')

plt.bar(hubei_df.Date[:18], under_report_pcnt)

plt.xticks(rotation=90)

plt.show()