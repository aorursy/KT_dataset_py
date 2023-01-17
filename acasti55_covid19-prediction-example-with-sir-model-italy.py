!pip install lmfit
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import random

import seaborn as sns

import matplotlib.dates as mdates

from matplotlib import dates

from sklearn.metrics import mean_squared_error, r2_score

from scipy.integrate import odeint

from lmfit import minimize, Parameters, Parameter, report_fit

from datetime import datetime



import dateutil.relativedelta as relativedelta
url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'

dfr = pd.read_csv(url,error_bad_lines=False)

dfr['data'] = pd.to_datetime(dfr['data'])

dfr['Days'] = (dfr.data.diff().dt.days+dfr.data.diff().dt.seconds/86400).cumsum().fillna(0).round(0)



today = pd.Timestamp(datetime.today().date()- relativedelta.relativedelta(days=1))
dfr['NuoviCasi Totali'] = dfr['totale_casi'].diff()

dfr['Hosp_R'] = dfr['totale_ospedalizzati']/dfr['totale_casi']

dfr['IC_R'] = dfr['terapia_intensiva']/dfr['totale_casi']

# Ratio of dead people over total cases

dfr['DR'] = dfr['deceduti']/dfr['totale_casi']

dfr[['data','terapia_intensiva','totale_attualmente_positivi','totale_casi','deceduti','Days','NuoviCasi Totali']].tail(3)
fig, ax = plt.subplots(figsize=(10,6))

fig.autofmt_xdate()

sns.lineplot(x='data',y='Hosp_R', data=dfr, label='Hospitalized', ax=ax, lw=2)

sns.lineplot(x='data',y='IC_R', data=dfr, label='Intensive Care', ax=ax, lw=2)

sns.lineplot(x='data',y='DR', data=dfr, label='Fatalities', ax=ax, lw=2)

plt.ylabel('Fraction', fontsize=14)

plt.xlabel('Date', fontsize=14)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())

ax.xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.legend(fontsize=12)

plt.title('Covid-19 Dead and Intensive care Ratios in Italy from 25/02/2020', fontsize=16);
def deriv(y, t, N, ps):

    S, I, R = y

    try:

        beta_i = ps['beta_i'].value

        tau = ps['tau'].value

        gamma = ps['gamma'].value

    except:

        beta_i, beta_l, tau, gamma = ps

    

    beta = beta_i*(1.1-tau*t)

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt



def odesol(y,t,N,ps):

    I0 = ps['i0'].value

    y0 = S0, I0, R0

    x = odeint(deriv, y0, t, args=(N, ps))

    return x

    

def residual(ps, ts, data):

    model = pd.DataFrame(odesol(y0,t,N,ps), columns=['S','I','R'])

    return (model['I'].values - data).ravel()


# Total population, N.

N = 10e6

# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 100, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0

# Initial conditions vector

y0 = S0, I0, R0



t = np.linspace(0, dfr.shape[0]-1, dfr.shape[0])



# set parameters incluing bounds

params = Parameters()

params.add('i0', value=I0, min=200, max=400)

params.add('beta_i', value= 0.35, min=0.1, max=0.5)

params.add('gamma', value= 0.11, min=0.09, max=0.12)

params.add('tau', value= 0.021, min=0.015, max=0.025)



#real data

data = dfr['totale_attualmente_positivi'].values

#model = pd.DataFrame(odesol(y0,t,N,ps), columns=['S','I','R'])



# fit model and find predicted values

result = minimize(residual, params, args=(t, data), method='leastsq')

final = data + result.residual.reshape(data.shape)

# plot data and fitted curves

plt.plot(t, data, 'o',c='k', label='Total Actual positives')

plt.plot(t, final, '--', linewidth=2, c='red', label='Best Fit ODE - SIR');

#plt.yscale('log')

plt.xlabel('Days')

plt.ylabel('Infected');
result
def deriv_betarolling(y, t, N, beta, gamma,cf):

    S, I, R = y

    beta = beta*(1.10-cf*t)

    if beta <0:

        beta=0

        

    dSdt = -beta * S * I / N

    dIdt = beta * S * I / N - gamma * I

    dRdt = gamma * I

    return dSdt, dIdt, dRdt

# Total population, N.

N = 60e6

# Initial number of infected and recovered individuals, I0 and R0.

I0, R0 = 286, 0

# Everyone else, S0, is susceptible to infection initially.

S0 = N - I0 - R0

# Initial conditions vector

y0 = S0, I0, R0



t = np.linspace(0, 70, 71)

beta_i = 0.36250283

gamma0 =  0.11600146

cf = 0.01927108

SIR = pd.DataFrame(odeint(deriv_betarolling, y0, t, args=(N, beta_i, gamma0,cf)), columns=['S','I','R'])

SIR['Days'] = t



gamma0 = 0.12

cf = 0.02025

SIRlow = pd.concat([SIR,pd.DataFrame(odeint(deriv_betarolling, y0, t, args=(N, beta_i, gamma0,cf)), columns=['S_low','I_low','R_low'])],axis=1)



gamma0 = 1/10

cf = 0.01975

SIRup = pd.concat([SIRlow,pd.DataFrame(odeint(deriv_betarolling, y0, t, args=(N, beta_i, gamma0,cf)), columns=['S_sup','I_sup','R_sup'])],axis=1)



SIRup['date'] =pd.date_range(dfr.data.min(), periods=SIRup.shape[0], freq='D')

SIRup['Tot'] = SIRup['I']+SIRup['R']

SIRup['Tot_low'] = SIRup['I_low']+SIRup['R_low']

SIRup['Tot_sup'] = SIRup['I_sup']+SIRup['R_sup']

label_std = 'Sim.'+r' $ \beta $ = $ \beta $*(1.1-0.0204), $\gamma$ = 0.1'

label_low = 'Sim.'+r' $ \beta $ = $ \beta $*(1.1-0.02),$\gamma$ = 0.12'

t2=SIRup.Days

I2=SIRup.I
SIRup[['I','R','Tot']].plot()

plt.title('Trend for infected (I), recovered (R) and total infected (Tot) \n Italian pop. (60M)')

plt.ylabel('N. People')

plt.xlabel('Days');
fig, ax = plt.subplots(1,2,figsize=(14,5))

fig.autofmt_xdate()



sns.lineplot(x='date',y=SIRup.I,data=SIRup, label=label_std, color='red', ax=ax[0]);

ax[0].scatter(x=dfr['data'], y=dfr['totale_attualmente_positivi'], label='Act. positives (Prot.Civ.)')

ax[0].set(xticks=pd.date_range(dfr.data.min(), periods=SIRup.shape[0]/7, freq='W'))

ax[0].xaxis.set_major_formatter(dates.DateFormatter("%d-%b-%Y"))

ax[0].legend()

ax[0].set_title('Sim. actual positives vs official');

ax[0].set_ylabel('Numero Infetti')



sns.lineplot(x='date',y=SIRup.Tot,data=SIRup, label='Sim. Tot', color='red' ,ax=ax[1]);

sns.lineplot(x='date',y=SIRup.Tot_low,data=SIRup, label=label_low, color='green' ,ax=ax[1]);

ax[1].scatter(x=dfr['data'], y=dfr['totale_casi'], label='Tot official (Prot.Civ.)')

ax[1].set(xticks=pd.date_range(dfr.data.min(), periods=SIRup.shape[0]/7, freq='W'))

ax[1].xaxis.set_major_formatter(dates.DateFormatter("%d-%b-%Y"))

ax[1].set_ylabel('Total Infected')

ax[1].legend()



plt.title('Sim. Total infected vs official data');
dfr['Ext_from_fatalities'] = round(dfr['deceduti']/0.054,0)

dfr['Ext_from_hospitalized'] = round(dfr['totale_ospedalizzati']/0.25,0)

dfr['Ext_from_IC'] = round(dfr['terapia_intensiva']/0.03,0)

#dfr.head()
dfr[['totale_casi','Ext_from_fatalities','Ext_from_hospitalized','Ext_from_IC']].plot()

plt.yscale('log')

plt.xlabel('Days')

plt.ylabel('Totale casi');
dfr[['totale_casi','Ext_from_fatalities','Ext_from_hospitalized','Ext_from_IC']].plot()

plt.xlabel('Days')

plt.ylabel('Totale casi');
dfr[['data','deceduti','totale_ospedalizzati','totale_casi','Ext_from_fatalities','Ext_from_hospitalized','Ext_from_IC']].tail(2)