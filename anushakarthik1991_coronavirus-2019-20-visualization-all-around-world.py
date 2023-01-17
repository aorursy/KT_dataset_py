# Importing packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams.update({'font.size': 12})



# Load data

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])



print (data.shape)

print ('Last update: ' + str(data.ObservationDate.max()))
# To check every place has only one observation per day

checkdup = data.groupby(['Country/Region','Province/State','ObservationDate']).count().iloc[:,0]

checkdup[checkdup>1]
# Clean up rows with zero cases

data = data[data.Confirmed>0]

data.shape
# Checking where the duplicates come from

data[data['Province/State'].isin(['Hebei','Gansu']) & (data['ObservationDate'].isin(['2020-03-11','2020-03-12']))]
# Clean data

# data = data.drop([4926,4927,5147, 5148]) # Remove duplicates

# data.loc[data['Province/State']=='Macau', 'Country/Region'] = 'Macau' # March 14 - clean data for Macau and HK

# data.loc[data['Province/State']=='Hong Kong', 'Country/Region'] = 'Hong Kong'

data = data.drop(['SNo', 'Last Update'], axis=1)

data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})

# To check null values

data.isnull().sum()
data = data.sort_values(['Date','Country','Province/State'])
def get_place(row):

    if row['Province/State'] == 'Hubei':

        return 'Hubei PRC'

    elif row['Country'] == 'Mainland China': 

        return 'Others PRC'

    else: return 'World'

    

data['segment'] = data.apply(lambda row: get_place(row), axis=1)
latest = data[data.Date == data.Date.max()]

print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))

print ('Total death cases: %.d' %np.sum(latest['Deaths']))

print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))
cty = latest.groupby('Country').sum()

cty['Death Rate'] = cty['Deaths'] / cty['Confirmed'] * 100

cty['Recovery Rate'] = cty['Recovered'] / cty['Confirmed'] * 100

cty['Active'] = cty['Confirmed'] - cty['Deaths'] - cty['Recovered']

cty.sort_values('Confirmed', ascending=False).head(10)
# fig, axes = plt.subplots(2, 2, figsize=(15, 10))

def plot_new(column, title):

    if column == 'Death Rate':

        _ = cty[cty.Deaths>=5].sort_values('Death Rate', ascending=False)

    else:

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



plt.figure(figsize=(9,23))

plt.subplot(411)

plot_new('Confirmed','Confirmed cases top 15 countries')

plt.subplot(412)

plot_new('Deaths','Death cases top 15 countries')

plt.subplot(413)

plot_new('Active','Active cases top 15 countries')

plt.subplot(414)

plot_new('Death Rate','Death rate for countries >=5 deaths')

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
evo3 = data.groupby(['Date','segment'])[['Confirmed','Deaths','Recovered']].sum()

evo3 = evo3.reset_index(level='segment')

evo3['Active'] = evo3['Confirmed'] - evo3['Deaths'] - evo3['Recovered']

evo3['Death Rate'] = evo3['Deaths'] / evo3['Confirmed'] * 100
fig, ax = plt.subplots(2, 2, figsize=(12,9))



def plot_evo(num, col, title):

    ax[num].plot(evo3.loc[evo3.segment=='Hubei PRC',col], lw=2, label='Hubei PRC')

    ax[num].plot(evo3.loc[evo3.segment=='Others PRC',col], lw=2, label='Others PRC')

    ax[num].plot(evo3.loc[evo3.segment=='World',col], lw=2, label='World')

    ax[num].set_title(title)

    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))

    ax[num].xaxis.set_major_formatter(months_fmt)

    ax[num].grid(True)

    

plot_evo((0,0), 'Confirmed', 'Confirmed cases')

plot_evo((0,1), 'Deaths', 'Death cases')

plot_evo((1,0), 'Active', 'Active cases')

plot_evo((1,1), 'Death Rate', 'Death rate')

plt.legend(bbox_to_anchor=(1.45, 2.2))

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

evo_cty('Iran')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Italy')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('France')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Germany')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Spain')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('UK')
fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('US')