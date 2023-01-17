import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import math

import functools

from scipy.optimize import minimize

import warnings



warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df.head()
df.info()
# The below class is an implementation of SIR model

class SIR():

    def __init__(self, beta, alpha, t1):

        self.beta = beta

        self.eps = 0.0001

        self.d = alpha + t1

        

    def recovered(self, t):

        return 1/(1+np.exp(-self.beta*(t-self.d)))

    

    def infected(self, t):

        return 1 - self.recovered(t) - math.exp(np.log(self.eps)*self.recovered(t))

    

    def susceptible(self, t):

        return math.exp(np.log(self.eps)*self.recovered(t))
duration = list(range(0,200))



model = SIR(0.2, 100, min(duration))



plt.figure(figsize=(8, 6))

plt.plot(duration, list(map(model.susceptible,duration)), label='Susceptible')

plt.plot(duration, list(map(model.recovered,duration)), label='Recovered/Deceased')

plt.plot(duration, list(map(model.infected,duration)), label='Infected', color='red')

plt.title('SIR model with beta=0.2 (Example)')

plt.xlabel('Days')

plt.ylabel('% of population')

plt.legend()
df['Date'] = pd.to_datetime(df['Date'])

df.sort_values(by=['Date'])

df['Days'] = df['Date'] - min(df['Date'])



# Converting date to elapsed days

df['Days'] = df['Days'].apply(lambda x: x.days) 

world_trend_region = df.groupby(['Days','Country/Region']).sum()[['ConfirmedCases','Fatalities']]

world_trend_region.reset_index(inplace=True)



# Normalizing with total world population

WORLD_POPULATION = 7.6*10**9

world_trend_region['ConfirmedCases'] = world_trend_region['ConfirmedCases'].apply(lambda x: x/WORLD_POPULATION)

world_trend_region['Fatalities'] = world_trend_region['Fatalities'].apply(lambda x: x/WORLD_POPULATION)



world_trend = world_trend_region.groupby('Days').sum()[['ConfirmedCases','Fatalities']]

world_trend.reset_index(inplace=True)
def cost(param, dataset=None):

    beta = param[0]

    alpha = param[1]

    sm = 0

    m = SIR(beta, alpha, min(dataset['Days']))

    for t,y in dataset[['Days','ConfirmedCases']].to_numpy():

        sm += (m.infected(t)-y)**2

    return sm/dataset['Days'].size
beta_rg = np.linspace(0,0.9,10)

alpha_rg = np.linspace(1,70,10)

BETA, ALPHA = np.meshgrid(beta_rg, alpha_rg)

C = []

for X,Y in zip(BETA,ALPHA):

    c = []

    for x,y in zip(X,Y):

        s = functools.partial(cost, dataset=world_trend)

        c.append(s([x,y]))

    C.append(c)

fig,ax=plt.subplots(1,1)

ax.set_xlabel('beta')

ax.set_ylabel('alpha')

ax.set_title('Cost')

cp = ax.contourf(BETA, ALPHA, C)

fig.colorbar(cp)
res = minimize(functools.partial(cost, dataset=world_trend), [0.1,65], method='L-BFGS-B', tol=1e-100)
model = SIR(res.x[0], res.x[1], min(world_trend['Days']))



plt.figure(figsize=(8, 6))

plt.plot(duration, list(map(model.infected,duration)), label='Infected(as per model)', color='red')

plt.scatter(world_trend['Days'].to_numpy(),world_trend['ConfirmedCases'].to_numpy(), color='grey', label='Confirmed Cases')

plt.title('SIR model with beta=%s and alpha=%s' % (res.x[0],res.x[1]))

plt.xlabel('Days')

plt.ylabel('% of population')

plt.legend()

axes = plt.gca()

ulimit_x = max(world_trend['Days']) + (max(world_trend['Days'])*0.5)

ulimit_y = max(world_trend['ConfirmedCases']) + (max(world_trend['ConfirmedCases'])*0.5)

axes.set_xlim([min(world_trend['Days']),ulimit_x])

axes.set_ylim([min(world_trend['ConfirmedCases']),ulimit_y])
peak_day = minimize(lambda x: -model.infected(x), [10], method='L-BFGS-B', tol=1e-100).x[0]

max_affected = model.infected(peak_day)

end_day = minimize(lambda x: model.infected(x), [peak_day+1], method='L-BFGS-B', tol=1e-100).x[0]

print('Day at which the infection is expected to be at its peak:', int(peak_day))

print('Number of infected people when the infection is at its peak: %s' % "{:.2%}".format(max_affected))

print('Day when the infection is projected to end:', int(end_day))
class DynamicSIR():

    

    def __init__(self, dataset, width=None):

        self.dataset = dataset

        self.min_t = min(self.dataset[self.dataset['ConfirmedCases'] > 0]['Days'])

        self.max_t = max(self.dataset[self.dataset['ConfirmedCases'] > 0]['Days'])

        self.frame_width = int((self.max_t - self.min_t)/(width or 6))

        self.params = pd.DataFrame(self.get_params(), columns=['T1','beta','alpha'])

    

    def get_params(self):

        cnt = self.min_t

        params = []

        while cnt <= self.max_t:

            if cnt + self.frame_width <= self.max_t:

                f_trend = self.dataset[cnt:cnt+self.frame_width]

            else:

                f_trend = self.dataset[cnt:]

            res = minimize(functools.partial(cost, dataset=f_trend), [0.1,2], method='L-BFGS-B', tol=1e-100)

            params.append((cnt,res.x[0],res.x[1]))

            cnt += 1

        return params

    

    def infected(self, t):

        T1 = max(self.params['T1'])

        s, wgt_s, beta, alpha = 0, 0, 0, 0

        for t1 in range(max(0,t-self.frame_width),t):

            d = self.params[self.params['T1'] == t1]

            if d.size == 0:

                break

            beta, alpha = float(d['beta']), float(d['alpha'])

            infec = SIR(beta, alpha, t1).infected(t)

            if infec > 0:

                s += (t+1)*infec

                wgt_s += (t+1)

        return s/wgt_s if wgt_s != 0 else 0
def get_model(data, width):

    return DynamicSIR(data, width)



def plot_outbreak(countries, width=None):

    i = 0

    f, axs = plt.subplots(len(countries),3,figsize=(20,30))

    f.tight_layout(pad=3.0)

    for country in countries:

        if country == 'World':

            data = world_trend

        else:

            data = world_trend_region[world_trend_region['Country/Region'] == country]

        dSir = get_model(data, width)

        t = [x for x in dSir.params['T1'].to_numpy() if x < max(dSir.params['T1'].to_numpy())-dSir.frame_width]

        beta = dSir.params[dSir.params['T1'].isin(t)]['beta'].to_numpy()

        alpha = dSir.params[dSir.params['T1'].isin(t)]['alpha'].to_numpy()

        ax = axs if len(axs) == 3 else axs[i]

        ax[0].plot(t,beta)

        ax[0].set_title('%s Outbreak Control - Measured with beta' % country)

        ax[0].set_xlabel('Days')

        ax[0].set_ylabel('beta')

        ax[1].plot(t,alpha)

        ax[1].set_title('%s Outbreak Control - Measured with alpha' % country)

        ax[1].set_xlabel('Days')

        ax[1].set_ylabel('alpha')

        max_t = max(dSir.params['T1'].to_numpy()) + dSir.frame_width

        min_t = min(dSir.params['T1'].to_numpy())

        ax[2].scatter(range(min_t, max_t), list(map(dSir.infected, range(min_t, max_t))), color='red', label='Model', alpha=0.7)

        ax[2].set_yscale('log')

        ax[2].plot(data['Days'].to_numpy(),data['ConfirmedCases'].to_numpy(), color='grey', label='Confirmed Cases')

        ax[2].set_title('%s Infection (log scale)' % country)

        ax[2].set_xlabel('Days')

        ax[2].set_ylabel('% of population')

        ax[2].legend()

        i += 1
plot_outbreak(['World','France','India','Italy','Spain','Germany','China','Korea, South','Japan','Iran'])