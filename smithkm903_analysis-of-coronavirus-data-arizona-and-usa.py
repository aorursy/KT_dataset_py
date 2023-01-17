# Importing packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams.update({'font.size': 14})



# Load data

data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates = ['ObservationDate','Last Update'])



print (data.shape)

print ('Last update: ' + str(data.ObservationDate.max()))
# To check every place has only one observation per day

checkdup = data.groupby(['Country/Region','Province/State','ObservationDate']).count().iloc[:,0]

checkdup[checkdup>1]
# Checking where the duplicates come from

data[data['Province/State'].isin(['Hebei','Gansu']) & (data['ObservationDate'].isin(['2020-03-11','2020-03-12']))]
# Clean data

data = data.drop([4926,4927,5147, 5148]) # March 14 - remove duplicates

data.loc[data['Province/State']=='Macau', 'Country/Region'] = 'Macau' # March 14 - clean data for Macau and HK

data.loc[data['Province/State']=='Hong Kong', 'Country/Region'] = 'Hong Kong'

data = data.drop(['SNo', 'Last Update'], axis=1)

data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})

# To check null values

data.isnull().sum()
daily = data.sort_values(['Date','Country','Province/State'])
def get_place(row):

    if row['Province/State'] == 'Arizona':

        return 'Arizona USA'

    elif row['Country'] == 'US': 

        return 'Others USA'

    else: return 'World'

    

daily['segment'] = daily.apply(lambda row: get_place(row), axis=1)
latest = daily[daily.Date == daily.Date.max()]
print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))

print ('Total death cases: %.d' %np.sum(latest['Deaths']))

print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))
segment1 = latest.groupby('segment').sum()

segment1['Death Rate'] = segment1['Deaths'] / segment1['Confirmed'] * 100

segment1['Recovery Rate'] = segment1['Recovered'] / segment1['Confirmed'] * 100

segment1
# Confirmed Cases USA

_ = latest.loc[latest.segment=='Others USA',['Province/State','Confirmed']].sort_values('Province/State', ascending=True)

plt.figure(figsize=(9,7))

sns.barplot('Confirmed', 'Province/State', data = _)

plt.title('Top confirmed cases USA ex-Arizona')

plt.yticks(fontsize=8)

plt.grid(axis='x')

plt.show()
# Death Cases China USA

_ = latest.loc[latest.segment=='Others USA',['Province/State','Deaths']].sort_values('Deaths', ascending=False)

_ = _[_.Deaths>0]

plt.figure(figsize=(9,7))

sns.barplot('Deaths', 'Province/State', data = _)

plt.title('Top Death cases USA')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
# Confirmed Cases World ex-USA

worldstat = latest[latest.segment=='World'].groupby('Country').sum()

_ = worldstat.sort_values('Confirmed', ascending=False).head(15)

plt.figure(figsize=(9,6))

sns.barplot(_.Confirmed, _.index)

plt.title('Top 15 Confirmed cases World ex-USA')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
# Death Cases World ex-USA

_ = worldstat.sort_values('Deaths', ascending=False)

_ = _[_.Deaths>=5]

plt.figure(figsize=(9,6))

sns.barplot(_.Deaths, _.index)

plt.title('Top Deaths cases World ex-USA (5 or above)')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
# Compare death rate across countries

_ = latest.groupby('Country')['Confirmed','Deaths'].sum().reset_index()

_['Death rate'] = _['Deaths'] / _['Confirmed'] * 100

_ = _.sort_values('Death rate', ascending=False)

death_cty = _[_['Deaths']>=5]

plt.figure(figsize=(9,6))

sns.barplot(death_cty['Death rate'], death_cty['Country'])

plt.title('Death Rate Comparison (>=5 Deaths)')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
death_cty
import matplotlib.dates as mdates

months = mdates.MonthLocator()

months_fmt = mdates.DateFormatter('%b-%e')



confirm = pd.pivot_table(daily.dropna(subset=['Confirmed']), 

                         index='Date', columns='segment', values='Confirmed', aggfunc=np.sum).fillna(method = 'ffill')

fig, ax = plt.subplots(figsize=(11,6))

ax.plot(confirm, marker='o')

plt.title('Confirmed Cases')

ax.legend(confirm.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

ax.grid(True)

plt.show()
death = pd.pivot_table(daily.dropna(subset=['Deaths']), 

                         index='Date', columns='segment', values='Deaths', aggfunc=np.sum).fillna(method = 'ffill')

fig, ax = plt.subplots(figsize=(11,6))

ax.plot(death, marker='o')

plt.title('Death Cases')

ax.legend(death.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

ax.grid(True)

plt.show()
good = pd.pivot_table(daily.dropna(subset=['Recovered']), 

                         index='Date', columns='segment', values='Recovered', aggfunc=np.sum).fillna(method = 'ffill')

fig, ax = plt.subplots(figsize=(11,6))

ax.plot(good, marker='o')

plt.title('Recovered Cases')

ax.legend(good.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

ax.grid(True)

plt.show()
# Active case - confirmed minus deaths and recovered

daily['Active'] = daily['Confirmed'] - daily['Deaths'] - daily['Recovered']

active = pd.pivot_table(daily.dropna(subset=['Active']), 

                         index='Date', columns='segment', values='Active', aggfunc=np.sum).fillna(method = 'ffill')

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(active, marker='o')

plt.title('Active Cases')

ax.legend(active.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

ax.grid(True)

plt.show()
# Global ex-China - Top 10 countries

c10 = worldstat.sort_values('Confirmed', ascending=False).head(10).index.tolist()

# Confirmed cases

c10cases = daily[daily['Country'].isin(c10)]

confirm_w = pd.pivot_table(c10cases.dropna(subset=['Confirmed']), index='Date', 

                         columns='Country', values='Confirmed', aggfunc=np.sum).fillna(method='ffill')

fig, ax = plt.subplots(figsize=(12,7))

plt.plot(confirm_w[confirm_w.index>'2020-02-01'], marker='o')

plt.title('Confirmed Cases - Top 10 Countries Outside China')

ax.legend(confirm_w.columns, loc=2)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.show()
# Death cases

death_w = pd.pivot_table(c10cases.dropna(subset=['Deaths']), index='Date', 

                         columns='Country', values='Deaths', aggfunc=np.sum).fillna(method='ffill')

fig, ax = plt.subplots(figsize=(12,7))

plt.plot(death_w[death_w.index>'2020-02-01'], marker='o')

plt.title('Death Cases - Top 10 Countries Outside China')

plt.legend(death_w.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.show()
# Calculate death and recovery rate



df = confirm.join(death, lsuffix='_confirm', rsuffix='_death')

df = df.join(good.add_suffix('_recover'))

df['Arizona USA_death_rate'] = df['Arizona USA_death']/df['Arizona USA_confirm']

df['Others USA_death_rate'] = df['Others USA_death']/df['Others USA_confirm']

df['World_death_rate'] = df['World_death']/df['World_confirm']

df['Arizona USA_recover_rate'] = df['Arizona USA_recover']/df['Arizona USA_confirm']

df['Others USA_recover_rate'] = df['Others USA_recover']/df['Others USA_confirm']

df['World_recover_rate'] = df['World_recover']/df['World_confirm']
death_rate = df[['Arizona USA_death_rate','Others USA_death_rate','World_death_rate']]*100

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(death_rate, marker='o')

plt.title('Death Rate %')

plt.legend(death.columns)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.show()
recover_rate = df[['Arizona USA_recover_rate','Others USA_recover_rate','World_recover_rate']]*100

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(recover_rate, marker='o')

plt.title('Recovery Rate %')

plt.legend(good.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.show()