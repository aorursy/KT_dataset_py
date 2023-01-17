# Importa pacotes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet 
from datetime import datetime
from fbprophet import Prophet
dias = 5

plt.rcParams.update({'font.size': 12})

# Load data
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])

print (data.shape)
print ('Last update: ' + str(data.ObservationDate.max()))
# Para verificar todos os locais, há apenas uma observação por dia
checkdup = data.groupby(['Country/Region','Province/State','ObservationDate']).count().iloc[:,0]
checkdup[checkdup>1]
# EUA mostram casos recuperados em uma linha separada
data[(data['Country/Region']=='US') & (data['Province/State'] == 'Recovered')].tail()
# Lipando Dados
data = data[(data.Confirmed>0) | (data['Province/State'] == 'Recovered')]
data = data.drop(['SNo', 'Last Update'], axis=1)
data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})
# To check null values
data.isnull().sum()
# Classificar dados
data = data.sort_values(['Date','Country','Province/State'])
# Add column of days since first case
data['first_date'] = data.groupby('Country')['Date'].transform('min')
data['days'] = (data['Date'] - data['first_date']).dt.days
latest = data[data.Date == data.Date.max()]
print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))
print ('Total death cases: %.d' %np.sum(latest['Deaths']))
print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))
print ('Death rate %%: %.2f' % (np.sum(latest['Deaths'])/np.sum(latest['Confirmed'])*100))
cty = latest.groupby('Country').sum()
cty['Death Rate'] = cty['Deaths'] / cty['Confirmed'] * 100
cty['Recovery Rate'] = cty['Recovered'] / cty['Confirmed'] * 100
cty['Active'] = cty['Confirmed'] - cty['Deaths'] - cty['Recovered']
cty.drop('days',axis=1).sort_values('Confirmed', ascending=False).head(3)
def plot_new(column, title):
    _ = cty.sort_values(column, ascending=False).head(15)
    g = sns.barplot(_[column], _.index)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    for i, v in enumerate(_[column]):
        if column == 'Death Rate':
            g.text(v*1.01, i+0.1, str(round(v,2)))
        else:
            g.text(v*1.01, i+0.1, str(int(v)))

plt.figure(figsize=(9,16))
plt.subplot(311)
plot_new('Confirmed','Confirmed cases top 15 countries')
plt.subplot(312)
plot_new('Deaths','Death cases top 15 countries')
plt.subplot(313)
plot_new('Active','Active cases top 15 countries')

plt.show()
def plot_rate(rank, title):
    if rank == 'highest':
        _ = cty[cty.Deaths>=10].sort_values('Death Rate', ascending=False).head(15)
    else:
        _ = cty[cty.Confirmed>=500].sort_values('Death Rate').head(15)
    g = sns.barplot(_['Death Rate'], _.index)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    for i, v in enumerate(_['Death Rate']):
        g.text(v*1.01, i+0.1, str(round(v,2)))

plt.figure(figsize=(9,12))
plt.subplot(211)
plot_rate('highest','Highest death rate top 15 (>=10 deaths only)')
plt.subplot(212)
plot_rate('lowest','Lowest death rate top 15 (>=500 confirmed only)')
import matplotlib.dates as mdates
months_fmt = mdates.DateFormatter('%b-%e')

evo = data.groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
evo['Active'] = evo['Confirmed'] - evo['Deaths'] - evo['Recovered']
evo['Death Rate'] = evo['Deaths'] / evo['Confirmed'] * 100
evo['Recover Rate'] = evo['Recovered'] / evo['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))

def plot_evo(num, col, title):
    ax[num].plot(evo[col], lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)
    
plot_evo((0,0), 'Confirmed', 'Confirmed cases')
plot_evo((0,1), 'Deaths', 'Death cases')
plot_evo((1,0), 'Active', 'Active cases')
plot_evo((1,1), 'Death Rate', 'Death rate')

plt.show()
def plot_cty(num, evo_col, title):
    ax[num].plot(evo_col, lw=3)
    ax[num].set_title(title)
    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))
    ax[num].xaxis.set_major_formatter(months_fmt)
    ax[num].grid(True)

def evo_cty(country):
    evo_cty = data[data.Country==country].groupby('Date')[['Confirmed','Deaths','Recovered']].sum()
    evo_cty['Active'] = evo_cty['Confirmed'] - evo_cty['Deaths'] - evo_cty['Recovered']
    evo_cty['Death Rate'] = evo_cty['Deaths'] / evo_cty['Confirmed'] * 100
    plot_cty((0,0), evo_cty['Confirmed'], 'Confirmed cases')
    plot_cty((0,1), evo_cty['Deaths'], 'Death cases')
    plot_cty((1,0), evo_cty['Active'], 'Active cases')
    plot_cty((1,1), evo_cty['Death Rate'], 'Death rate')
    fig.suptitle(country, fontsize=16)
    plt.show()
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Mainland China')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Hong Kong')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Taiwan')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Japan')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('South Korea')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Singapore')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Iran')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Italy')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Spain')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('UK')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Germany')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('US')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Russia')
fig, ax = plt.subplots(2, 2, figsize=(12,9))
evo_cty('Brazil')
pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv',
                 usecols = ['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)'])
pop.columns = ['Country','Population','Density']
# Clean up country names
to_replace = ['China','United States','DR Congo', 'United Kingdom','Myanmar','Côte d\'Ivoire', 'Czech Republic (Czechia)',
             'Congo','Macao','State of Palestine','St. Vincent & Grenadines', 'Saint Kitts & Nevis']
replace_by = ['Mainland China','US','Congo (Kinshasa)','UK','Burma','Ivory Coast','Czech Republic',
             'Congo (Brazzaville)','Macau','West Bank and Gaza','Saint Vincent and the Grenadines', 'Saint Kitts and Nevis']
pop.replace(to_replace, replace_by, inplace=True)
pop.head()
cty_p = cty.reset_index()
cty_p = cty_p.merge(pop, how='left', on = 'Country')
cty_p.head()
# Check missing countries
nas = cty_p[cty_p.isnull().any(axis=1)]
nas[['Country','Confirmed','Population']]
cty_p['Case per M pop'] = cty_p['Confirmed'] / cty_p['Population'] * 1000000
cty_p['Death per M pop'] = cty_p['Deaths'] / cty_p['Population'] * 1000000
def plot_pop_case(rank, title):
    if rank == 'highest':
        _ = cty_p.sort_values('Case per M pop', ascending=False).head(15)
    else:
        _ = cty_p.sort_values('Case per M pop').head(15)
    g = sns.barplot(_['Case per M pop'], _.Country)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    if rank == 'highest':
        for i, v in enumerate(_['Case per M pop']):
            g.text(v*1.01, i+0.1, str(int(v)))
    else:
        for i, v in enumerate(_['Case per M pop']):
            g.text(v*1.01, i+0.1, str(round(v,2)))

plt.figure(figsize=(9,12))
plt.subplot(211)
plot_pop_case('highest','Countries with most cases per million population')
plt.subplot(212)
plot_pop_case('lowest','Countries with fewest cases per million population')
def plot_pop_death(rank, title):
    if rank == 'highest':
        _ = cty_p.sort_values('Death per M pop', ascending=False).head(15)
    else:
        _ = cty_p[cty_p.Confirmed>=500].sort_values('Death per M pop').head(15)
    g = sns.barplot(_['Death per M pop'], _.Country)
    plt.title(title, fontsize=14)
    plt.ylabel(None)
    plt.xlabel(None)
    plt.grid(axis='x')
    if rank == 'highest':
        for i, v in enumerate(_['Death per M pop']):
            g.text(v*1.01, i+0.1, str(int(v)))
    else:
        for i, v in enumerate(_['Death per M pop']):
            g.text(v*1.01, i+0.1, str(round(v,2)))

plt.figure(figsize=(9,12))
plt.subplot(211)
plot_pop_death('highest','Countries with most deaths per million population')
plt.subplot(212)
plot_pop_death('lowest','Countries with fewest deaths per million population (>=100 confirmed only)')
mortes = data.groupby('Country').sum()['Deaths'].reset_index()
mortes.tail(dias)
mortes.columns = ['ds','y']
mortes.tail(dias)
m = Prophet(interval_width=0.95)
m.fit(mortes)
futuro = m.make_future_dataframe(periods=dias)
futuro.tail(dias)
previsao = m.predict(futuro)
previsao.tail(dias)
previsao[['ds','yhat_lower','yhat','yhat_upper']].tail(dias)
confirmed_forecast_plot = m.plot(previsao)
confirmed_forecast_plot = m.plot(previsao)