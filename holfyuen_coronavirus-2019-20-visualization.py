# Importing packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime, timedelta



plt.rcParams.update({'font.size': 12})



# Load data

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])



print (data.shape)

print ('Last update: ' + str(data.ObservationDate.max()))
# Clean data

data = data[(data.Confirmed>0) | (data['Province/State'] == 'Recovered') | (data.Recovered > 0)]

data = data.drop(['SNo', 'Last Update'], axis=1)

data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})

# To check null values

data.isnull().sum()
# Sort data

data = data.sort_values(['Date','Country','Province/State'])

# Add column of days since first case

data['first_date'] = data.groupby('Country')['Date'].transform('min')

data['days'] = (data['Date'] - data['first_date']).dt.days
last_date = data.Date.max()

date_minus7 = data.Date.max() + timedelta(days=-7)

latest = data[data.Date == last_date]

past_7 = data[data.Date == date_minus7]



c0 = np.sum(latest['Confirmed'])

c7 = np.sum(past_7['Confirmed'])

c_pct = (c0-c7)/c7 * 100

print ('Total confirmed cases %.d, +%.d or %.2f%% from 7 days ago.' % (c0, c7, c_pct))
d0 = np.sum(latest['Deaths'])

d7 = np.sum(past_7['Deaths'])

d_pct = (d0-d7)/d7 * 100

print ('Total death cases %.d, +%.d or %.2f%% from 7 days ago.' %(d0, d7, d_pct))
a0 = np.sum(latest['Confirmed']) - np.sum(latest['Deaths']) - np.sum(latest['Recovered'])

a7 = np.sum(past_7['Confirmed']) - np.sum(past_7['Deaths']) - np.sum(past_7['Recovered'])

a_pct = (a0-a7)/a7 * 100

print ('Total active cases: %.d, compared to %.d 7 days ago. Change %.2f%%.' %(a0, a7, a_pct))
dr0 = d0/c0 * 100

dr7 = d7/c7 * 100

print ('Death rate: %.2f%%, compared to %.2f%% 7 days ago' % (dr0, dr7))
cty = latest.groupby('Country').sum()

cty['Death Rate'] = cty['Deaths'] / cty['Confirmed'] * 100

cty['Recovery Rate'] = cty['Recovered'] / cty['Confirmed'] * 100

cty['Active'] = cty['Confirmed'] - cty['Deaths'] - cty['Recovered']



# Table of 20 countries with most cases

_ = cty.drop(['days','Death Rate', 'Recovery Rate'],axis=1).sort_values('Confirmed', ascending=False).head(20)

_.astype('int64').style.background_gradient(cmap='Reds')
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

        _ = cty[cty.Confirmed>=100].sort_values('Death Rate').head(15)

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

plot_rate('lowest','Lowest death rate top 15 (>=100 confirmed only)')
cty7 = past_7.groupby('Country').sum()

cty7['Death Rate'] = cty7['Deaths'] / cty7['Confirmed'] * 100

cty7['Recovery Rate'] = cty7['Recovered'] / cty7['Confirmed'] * 100

cty7['Active'] = cty7['Confirmed'] - cty7['Deaths'] - cty7['Recovered']
cty7 = cty7.add_suffix('_7')

cty7.head()
cty = pd.concat([cty, cty7], axis=1)

cty['new_confirm'] = cty['Confirmed'] - cty['Confirmed_7']

cty['new_deaths'] = cty['Deaths'] - cty['Deaths_7']

cty['active_chg'] = cty['Active'] - cty['Active_7']
def plot_change(column, asc, title):

    _ = cty.sort_values(column, ascending=asc).head(15)

    g = sns.barplot(_[column], _.index)

    plt.title(title)

    plt.ylabel(None)

    plt.xlabel(None)

    plt.grid(axis='x')

    for i, v in enumerate(_[column]):

        g.text(v*1.01, i+0.1, str(int(v)))



plt.figure(figsize=(9,20))

plt.subplot(411)

plot_change('new_confirm', False, 'Most new confirmed cases last 7 days')

plt.subplot(412)

plot_change('new_deaths', False, 'Most new death cases last 7 days')

plt.subplot(413)

plot_change('active_chg', False, 'Biggest increase in active cases last 7 days')

plt.subplot(414)

plot_change('active_chg', True, 'Biggest decrease in active cases last 7 days')



plt.show()
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

evo_cty('Vietnam')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('India')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Pakistan')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Australia')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Italy')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Spain')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('UK')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Germany')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('France')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Belgium')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Switzerland')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Sweden')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Serbia')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Russia')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('US')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Canada')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Mexico')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Brazil')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Argentina')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Chile')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Peru')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Iran')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Saudi Arabia')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Qatar')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Turkey')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Yemen')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Bahrain')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('South Africa')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Ethiopia')
pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv',

                 usecols = ['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)'])

pop.columns = ['Country','Population','Density']

# Clean up country names

to_replace = ['China','United States','DR Congo', 'United Kingdom','Myanmar','Côte d\'Ivoire', 'Czech Republic (Czechia)',

             'Congo','Macao','State of Palestine','St. Vincent & Grenadines', 'Saint Kitts & Nevis']

replace_by = ['Mainland China','US','Congo (Kinshasa)','UK','Burma','Ivory Coast','Czech Republic',

             'Congo (Brazzaville)','Macau','West Bank and Gaza','Saint Vincent and the Grenadines', 'Saint Kitts and Nevis']

pop.replace(to_replace, replace_by, inplace=True)
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

        _ = cty_p[cty_p.Population>1000000].sort_values('Case per M pop', ascending=False).head(15)

    else:

        _ = cty_p[cty_p.Population>1000000].sort_values('Case per M pop').head(15)

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

plot_pop_case('highest','Countries with most cases per million population\n(>1M pop only)')

plt.subplot(212)

plot_pop_case('lowest','Countries with fewest cases per million population\n(>1M pop only)')
def plot_pop_death(rank, title):

    if rank == 'highest':

        _ = cty_p[cty_p.Population>1000000].sort_values('Death per M pop', ascending=False).head(15)

    else:

        _ = cty_p[cty_p.Confirmed>=100].sort_values('Death per M pop').head(15)

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

plot_pop_death('highest','Countries with most deaths per million population (>1M pop only)')

plt.subplot(212)

plot_pop_death('lowest','Countries with fewest deaths per million population (>=100 confirmed only)')