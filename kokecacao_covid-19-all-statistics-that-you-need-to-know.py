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

data = data.drop([4926, 4927, 5147, 5148]) # March 14 - remove duplicates

data.loc[data['Province/State']=='Macau', 'Country/Region'] = 'Macau' # March 14 - clean data for Macau and HK

data.loc[data['Province/State']=='Hong Kong', 'Country/Region'] = 'Hong Kong'

data = data.drop(['SNo', 'Last Update'], axis=1)

data = data.rename(columns={'Country/Region': 'Country', 'ObservationDate':'Date'})

# To check null values

data.isnull().sum()
daily = data.sort_values(['Date','Country','Province/State'])
def get_place(row):

    if row['Province/State'] == 'Hubei':

        return 'Hubei PRC'

    elif row['Country'] == 'Mainland China': 

        return 'Others PRC'

    elif row['Country'] == 'US':

        return 'US'

    elif row['Country'] in ['Italy','Spain','Germany','France','Switzerland','Norway','UK','Netherlands','Sweden','Belgium','Denmark']:

        return 'Europe'

    elif row['Country'] == 'Iran':

        return 'Iran'

    elif row['Country'] == 'South Korea':

        return 'South Korea'

    else: return 'Rest of World'



daily['segment'] = daily.apply(lambda row: get_place(row), axis=1)
latest = daily[daily.Date == daily.Date.max()]
print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))

print ('Total death cases: %.d' %np.sum(latest['Deaths']))

print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))
segment1 = latest.groupby('segment').sum()

segment1['Death Rate'] = segment1['Deaths'] / segment1['Confirmed'] * 100

segment1['Recovery Rate'] = segment1['Recovered'] / segment1['Confirmed'] * 100

segment1
# Confirmed Cases World ex-China

worldstat = latest.groupby('Country').sum()

_ = worldstat.sort_values('Confirmed', ascending=False).head(15)

plt.figure(figsize=(9,6))

sns.barplot(_.Confirmed, _.index)

plt.title('Top 15 Confirmed cases World')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
# Death Cases World ex-China

_ = worldstat.sort_values('Deaths', ascending=False)

_ = _[_.Deaths>=10]

plt.figure(figsize=(9,6))

sns.barplot(_.Deaths, _.index)

plt.title('Top Deaths cases World (10 or above)')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
# Compare death rate across countries

_ = latest.groupby('Country')['Confirmed','Deaths'].sum().reset_index()

_['Death rate'] = _['Deaths'] / _['Confirmed'] * 100

_ = _.sort_values('Death rate', ascending=False)

death_cty = _[_['Deaths']>=10]

plt.figure(figsize=(9,6))

sns.barplot(death_cty['Death rate'], death_cty['Country'])

plt.title('Death Rate Comparison (>=10 Deaths)')

plt.yticks(fontsize=12)

plt.grid(axis='x')

plt.show()
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
# Active case - confirmed minus deaths and recovered

daily['Active'] = daily['Confirmed'] - daily['Deaths'] - daily['Recovered']

active = pd.pivot_table(daily.dropna(subset=['Active']), 

                         index='Date', columns='segment', values='Active', aggfunc=np.sum).fillna(method = 'ffill')

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(active, marker='o')

plt.title('ln(x) scale of active cases')

ax.legend(active.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

ax.set_yscale('log', basey=2.718) # LOG

plt.xticks(rotation=45, fontsize=12)

ax.grid(True)

# plt.ylim((0, 10000)) # SHOW US

plt.show()
n_future = 5

n_poly = 4



_ = list(active['US'])[30:]

_ = [x / 327200000*1000000 for x in _]

_ = _[3:]

p = np.polyfit(range(len(_)), _, n_poly)

f = np.polyval(p,range(len(_)+n_future))

plt.title("Percent Active per Million in US")

plt.xlabel("days after 1/million people infected")

plt.ylabel("active cases per million people")

plt.plot(range(len(_)),_,'o')

plt.plot(range(len(_)+n_future),f,'-')
_ = list(active['Rest of World'])

_ = [x / (7700000000-1386000000-58500000-741400000)*1000000 for x in _]

_ = _[25:]

p = np.polyfit(range(len(_)), _, n_poly)

f = np.polyval(p,range(len(_)+n_future))

plt.title("Percent Active per Million in Rest of World")

plt.xlabel("days after 1/million people infected")

plt.ylabel("active cases per million people")

plt.plot(range(len(_)),_,'o')

plt.plot(range(len(_)+n_future),f,'-')
_ = list(active['Europe'])

_ = [x / 741400000*1000000 for x in _]

_ = _[32:]

p = np.polyfit(range(len(_)), _, n_poly)

f = np.polyval(p,range(len(_)+n_future))

plt.title("Percent Active per Million in Europe")

plt.xlabel("days after 1/million people infected")

plt.ylabel("active cases per million people")

plt.plot(range(len(_)),_,'o')

plt.plot(range(len(_)+n_future),f,'-')
_ = list(active['Others PRC'])

_ = [x / (1386000000-58500000)*1000000 for x in _]

_ = _[6:]

p = np.polyfit(range(len(_)), _, n_poly)

f = np.polyval(p,range(len(_)+n_future))

plt.title("Percent Active per Million in China")

plt.xlabel("days after 1/million people infected")

plt.ylabel("active cases per million people")

plt.plot(range(len(_)),_,'o')

plt.plot(range(len(_)+n_future),f,'-')
_ = list(active['Hubei PRC'])

_ = [x / 58500000*1000000 for x in _]

_ = _[:]

p = np.polyfit(range(len(_)), _, n_poly)

f = np.polyval(p,range(len(_)+n_future))

plt.title("Percent Active per Million in Hubei")

plt.xlabel("days after 1/million people infected")

plt.ylabel("active cases per million people")

plt.plot(range(len(_)),_,'o')

plt.plot(range(len(_)+n_future),f,'-')
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

df['Hubei PRC_death_rate'] = df['Hubei PRC_death']/df['Hubei PRC_confirm']

df['Others PRC_death_rate'] = df['Others PRC_death']/df['Others PRC_confirm']

df['Rest of World_death_rate'] = df['Rest of World_death']/df['Rest of World_confirm']

df['US_death_rate'] = df['US_death']/df['US_confirm']

df['Europe_death_rate'] = df['Europe_death']/df['Europe_confirm']

df['Iran_death_rate'] = df['Iran_death']/df['Iran_confirm']

df['South Korea_death_rate'] = df['South Korea_death']/df['South Korea_confirm']



df['Hubei PRC_recover_rate'] = df['Hubei PRC_recover']/df['Hubei PRC_confirm']

df['Others PRC_recover_rate'] = df['Others PRC_recover']/df['Others PRC_confirm']

df['Rest of World_recover_rate'] = df['Rest of World_recover']/df['Rest of World_confirm']

df['US_recover_rate'] = df['US_recover']/df['US_confirm']

df['Europe_recover_rate'] = df['Europe_recover']/df['Europe_confirm']

df['Iran_recover_rate'] = df['Iran_recover']/df['Iran_confirm']

df['South Korea_recover_rate'] = df['South Korea_recover']/df['South Korea_confirm']
death_rate = df[['Hubei PRC_death_rate','Others PRC_death_rate','Rest of World_death_rate', 'US_death_rate', 'Europe_death_rate', 'Iran_death_rate', 'South Korea_death_rate']]*100

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(death_rate, marker='o')

plt.title('Death Rate %')

plt.legend(death.columns)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.ylim((0, 10)) # SHOW US

plt.show()
recover_rate = df[['Hubei PRC_recover_rate','Others PRC_recover_rate','Rest of World_recover_rate', 'US_recover_rate', 'Europe_recover_rate', 'Iran_recover_rate', 'South Korea_recover_rate']]*100

fig, ax = plt.subplots(figsize=(11,6))

plt.plot(recover_rate, marker='o')

plt.title('Recovery Rate %')

plt.legend(good.columns, loc=2, fontsize=12)

ax.xaxis.set_major_locator(plt.MaxNLocator(7))

ax.xaxis.set_major_formatter(months_fmt)

plt.xticks(rotation=45, fontsize=12)

plt.grid(True)

plt.show()
import datetime



# data from: https://www.immd.gov.hk/hkt/message_from_us/stat.html

plt.plot([339143, 229589, 174474, 144110, 127967, 124574, 107977, 97452, 94202, 81669, 86081, 52016, 52220, 63124, 57988, 29089, 27507, 24429, 19181, 20024, 21541, 22339, 20410, 19458, 17649, 15481, 16059, 19470, 20897, 18630, 18055, 17785, 15329, 15866, 17555, 20644, 20601], marker='o')

plt.title("HK Customer Data")

plt.xlabel("days after Jan.23")

plt.ylabel("number of people going out of HK")

plt.show()