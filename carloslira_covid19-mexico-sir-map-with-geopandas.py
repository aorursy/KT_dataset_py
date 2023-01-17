import numpy as np

import pandas as pd

import geopandas as gpd

import matplotlib.pyplot as plt 

plt.style.use('seaborn')



from scipy.integrate import odeint

from scipy import optimize
mexico = pd.read_csv('https://raw.githubusercontent.com/carloscerlira/COVIDMX/master/data/covidmx.csv')



plt.hist(mexico['Edad'], bins=20, range=(0,100), color='royalblue', histtype='bar', ec='black')

plt.xlabel('Edad')

plt.ylabel('Número de infectados')

plt.show()
df = pd.read_csv('https://raw.githubusercontent.com/carloscerlira/COVIDMX/master/data/covidworld.csv')

df = df.drop('SNo', axis=1)



def get_country(country_name):

    data = df[df['Country/Region']==country_name]

    data = data[data['Confirmed']>=100]

    country = data.groupby('ObservationDate')

    country = country.sum()

    return country



mexico = get_country('Mexico')['Confirmed']/129.2e6*1e6

days = mexico.size

print(days)

plt.plot(np.arange(1, days+1), mexico, linewidth=4, label='Mexico')



countries = ['US','France', 'Spain', 'Germany', 'Italy']

population = [327.2e6, 66.99e6, 46.66e6, 82.79e6, 60.48e6]

for i in range(0,len(countries)):

    country = get_country(countries[i])['Confirmed']/population[i]*1e6

    plt.plot(np.arange(1, days+1), country[:days], linewidth=1.5, label=countries[i]) 



plt.ylabel('Cases per 1 million')

plt.xlabel('Days since confirmed cases is greaten than 100')

plt.legend()

plt.show()
df = pd.read_csv('https://raw.githubusercontent.com/carloscerlira/COVIDMX/master/data/covidworld.csv') 

df = df.drop('SNo', axis=1)

df['Infected'] = df['Confirmed'] - df['Deaths'] - df['Recovered']



def deriv(y, t, beta, gamma):

    S, I, R = y

    dSdt = -beta*S*I/N

    dIdt = beta*S*I/N - gamma*I

    dRdt = gamma*I

    return dSdt, dIdt, dRdt

    

def sir(x, beta, gamma):

    I0, R0 = 1, 0

    S0 = N - I0 - R0

    t = np.linspace(0, 100, 500)

    y0 = S0, I0, R0

    ret = odeint(deriv, y0, t, args=(beta, gamma))

    S, I, R = ret.T

    return np.interp(x, t, I)



def forecast(country_name):

    data = df[(df['Country/Region'] == country_name)]

    gb = data.groupby('ObservationDate')

    country = gb.sum()

    country = country.reset_index()

    infected = country['Infected'].values

    plt.scatter(np.arange(0, infected.size, 1), infected, label='infected test', color='crimson', s=12) 



    y_data = country['Infected'].values[:-3]

    x_data = np.arange(0, y_data.size, 1)

    plt.scatter(x_data, y_data, label='infected train', color='royalblue', s=12) 



    params, params_covariance = optimize.curve_fit(sir, x_data, y_data, p0=[1,1])

    x_data = np.linspace(0, 80, 500)

    I0, R0 = 1, 0

    S0 = N - I0 - R0

    y0 = S0, I0, R0

    ret = odeint(deriv, y0, x_data, args=(params[0], params[1]))

    S, I, R = ret.T



    plt.plot(x_data, I, label='infected fit', color='black')

    plt.ylabel('number of infected persons')

    plt.xlabel('days')

    plt.legend()

    plt.show()
N = 120e6

forecast('Mexico')
N = 44.66e6 

forecast('Spain')
states = gpd.read_file('../input/covidmx/data/geopandas/states.shp')



plt.style.use('seaborn')

vmin = states['infected'].min(); vmax = states['infected'].max();

states.plot(column='infected', cmap='Blues', norm=plt.Normalize(vmin=vmin,vmax=vmax), linewidth=0.3, edgecolor='.8')



sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))

sm._A = []

plt.colorbar(sm)



plt.title('Casos confirmados COVID-19 \n 31/03/2020', fontdict={'fontsize': '18', 'fontweight' : '3'})

plt.annotate('Fuente: Plataforma COVID-19, SINVAE',xy=(0.15, .15),  xycoords='figure fraction', horizontalalignment='left', verticalalignment='top', fontsize=10, color='#555555')

plt.axis('off')

plt.show()