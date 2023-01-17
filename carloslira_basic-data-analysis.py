import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
plt.style.use('seaborn')
mexico = pd.read_csv('../input/covid19-mexico/confirmedraw.csv')
plt.hist(mexico['Edad'], bins=20, range=(0,100), color='royalblue', histtype='bar', ec='black')
plt.xlabel('Edad')
plt.ylabel('Número de infectados')
plt.show()
df = pd.read_csv('../input/covid19-mexico/time_series_confirmed_MX.csv', index_col=0)
df = df[:-1]
for state in df.index:
    state = df.loc[state]
    state = state[state >= 100]
    if(state.size==0):
        continue;
    state = state[:10]
    state.reset_index(inplace=True, drop=True)
    state.plot(label=state.name)
plt.xlabel('Days since number of confirmed cases reached 100')
plt.ylabel('Confirmed cases')
plt.legend()
plt.show()
fecha = '4/9/20'
confirmedmx = pd.read_csv('../input/covid19-mexico/time_series_confirmed_MX.csv', index_col=0)
deathsmx = pd.read_csv('../input/covid19-mexico/time_series_deaths_MX.csv', index_col=0)
death_rate = deathsmx[fecha].div(confirmedmx[fecha][confirmedmx['4/9/20'] > 100])*100
death_rate.dropna(inplace=True)
death_rate.sort_values(ascending=False, inplace=True)
death_rate.plot.bar(color='royalblue')
plt.title('Mortalidad \n (estados con más de 100 casos confirmados)')
plt.xlabel('')
plt.show()
populationmx = pd.read_csv('../input/covid19-mexico/population_MX.csv', index_col=0)
populationmx = populationmx['Población']
infected = confirmedmx[fecha][confirmedmx[fecha] > 100].div(populationmx)*1000000
infected.dropna(inplace=True)
infected.sort_values(ascending=False, inplace=True)
infected.plot.bar(color='royalblue')
plt.title('Casos confirmados por un millon de habitantes \n (estados con más de 100 casos confirmados)')
plt.xlabel('')
plt.show()
