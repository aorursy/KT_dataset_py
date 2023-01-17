

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

confirmed
confirmed = pd.melt(confirmed, id_vars=['Province/State','Country/Region'], value_vars=confirmed.columns[4:].values.tolist(), var_name='Date', value_name='Confirmed')

confirmed.Date = pd.to_datetime(confirmed.Date)

confirmed
ttl_confirmed = confirmed.groupby('Country/Region').Confirmed.max().sort_values(ascending=False)

ttl_confirmed.head(10)
confirmed_ts = confirmed.pivot_table(index='Date',columns='Country/Region',values='Confirmed', aggfunc=sum).loc[:,ttl_confirmed.head(10).index.values]

confirmed_ts
[print(t) for t in ttl_confirmed.head(10).index]
confirmed_ts.plot(figsize=(21,5))

plt.title('Top 10 Countries with Confirmed Cases');
death = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

death
death = pd.melt(death, id_vars=['Province/State','Country/Region'], value_vars=death.columns[4:].values.tolist(), var_name='Date', value_name='Deads')

death.Date = pd.to_datetime(death.Date)

death
ttl_death = death.groupby('Country/Region').Deads.max().sort_values(ascending=False)

ttl_death.head(10)
death_ts = death.pivot_table(index='Date',columns='Country/Region',values='Deads', aggfunc=sum).loc[:,ttl_death.head(10).index.values]

death_ts
death_ts.plot(figsize=(21,5))

plt.title('Top 10 Countries with Deaths');
recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

recovered
recovered = pd.melt(recovered, id_vars=['Province/State','Country/Region'], value_vars=recovered.columns[4:].values.tolist(), var_name='Date', value_name='Recovered')

recovered.Date = pd.to_datetime(recovered.Date)

display(recovered)
ttl_recovered = recovered.groupby('Country/Region').Recovered.max().sort_values(ascending=False)

ttl_recovered.head(10)
recovered_ts = recovered.pivot_table(index='Date',columns='Country/Region',values='Recovered', aggfunc=sum).loc[:,ttl_recovered.head(10).index.values]

recovered_ts.plot(figsize=(21,5))

plt.title('Top 10 Countries with Recovered Cases');
COUNTRY = 'China'
POPULATION = int(ttl_confirmed[COUNTRY]/.8) # assume 80 of total recovered

print(f'Population {POPULATION:,}')
train_death = death[(death['Country/Region']==COUNTRY)].groupby(['Date'])[['Deads']].sum().fillna(0)

train_recovered = recovered[(recovered['Country/Region']==COUNTRY)].groupby(['Date'])[['Recovered']].sum().fillna(0)
R_train = train_death.join(train_recovered)

R_train['R']=R_train.Deads + R_train.Recovered

R_train.drop(['Deads','Recovered'], axis=1, inplace=True)

R_train.index.freq='D'

#R_train = R_train.fillna(method='ffill')

R_train = R_train / POPULATION

R_train
plt.figure(figsize=(28,5))

plt.plot(R_train)

plt.grid()

plt.title(f'Total Removed in {COUNTRY} \n Total {R_train.R.max()*POPULATION:1,.0f}')

plt.show();
infected = confirmed[(confirmed['Country/Region']==COUNTRY)].groupby('Date')[['Confirmed']].sum().fillna(0)
infected.plot(figsize=(28,5))

plt.grid()

plt.title(f'Total Infections in {COUNTRY} \n Total {infected.max().values[0]:,}')

plt.ylabel('Confirmed Infected People');
I_train = infected / POPULATION
I_train.columns = ['I']

I_train.I = I_train.I - R_train.R

I_train
plt.figure(figsize=(28,5))

plt.plot(I_train)

plt.grid()

plt.legend()

plt.title(f'Infections in {COUNTRY} by Date \n as % of Susceptible People')

plt.ylabel('Infected People \n as % of Susceptible People');

plt.show();
from scipy.integrate import odeint

from scipy.optimize import minimize

import numpy as np

import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error
def SIR_mod(y, t, beta, gamma):

    S,I,R = y

    dS_dt = -beta * S * I

    dI_dt = beta * S * I - gamma * I

    dR_dt = gamma * I

    return([dS_dt,dI_dt,dR_dt])
I0=I_train.I[0]

S0=1.0-I_train.I[0]

R0=R_train.R[0]



beta=0.17

gamma=1/28 # where 28 days is the stimated days each person is infectious  



t=np.arange(0,R_train.shape[0])



solution = np.array(odeint(SIR_mod,[S0,I0,R0],t,args=(beta,gamma)))

display(solution[:5,:])

S, I, R = solution[:,0], solution[:,1], solution[:,2]
plt.figure(figsize=(21,7))

plt.plot(t, S, label='Susceptable')

plt.plot(t, I, label='Infected')

plt.plot(t, R, label='Recovered')

plt.plot(t, I_train.I.values, ':', label='Infected Observed')

plt.plot(t, R_train.R.values, ':', label='Recovered Observed')

plt.grid()

plt.legend()

plt.xlabel('Time')

plt.ylabel('Proportions')

plt.title('Generic SIR Model and Data to Fit')

plt.show();
def SIR_func(params):

    _beta, _gamma = params[0], params[1]

    sol = np.array(odeint(SIR_mod,[S0,I0,R0],t,args=(_beta,_gamma)))

    #_S, _I, _R = sol[:,0], sol[:,1], sol[:,2]

    return sol





def SIR_MSE(params):

    _SIR = SIR_func(params)

    return mean_squared_error(R_train.R.values,_SIR[:,2]) + mean_squared_error(I_train.values,_SIR[:,1])
res = minimize(fun=SIR_MSE, x0=[0.,0.])

res
solution = np.array(odeint(SIR_mod,[S0,I0,R0],t,args=(res.x[0],res.x[1])))

S, I, R = solution[:,0], solution[:,1], solution[:,2]
plt.figure(figsize=(21,7))

plt.plot(t, S, label='Susceptable Fitted')

plt.plot(t, I, label='Infected Fitted')

plt.plot(t, R, label='Recovered Fitted')

plt.plot(t,I_train.I.values,':',label='Infected Observed')

plt.plot(t,R_train.R.values,':',label='Recovered Observed')

plt.grid()

plt.legend()

plt.xlabel('Time')

plt.ylabel('Proportions')

plt.title('Fitted SIR Model for {COUNTRY}')

plt.show();
print(f'MSE:{SIR_MSE((res.x[0],res.x[1])):7.3f} for Estimated Beta:{res.x[0]:7.3f} and Gamma:{res.x[1]:7.3f}')

print(f'{COUNTRY} Expected duration of infection{1/res.x[1]:7.3f} days (1/Gamma)')

print(f'{COUNTRY} Estimated R0: {res.x[0]/res.x[1]:5.2f} (Beta/Gamma)')