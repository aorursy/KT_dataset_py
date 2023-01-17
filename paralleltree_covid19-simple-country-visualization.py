import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', parse_dates=['Date'])

df_train = df_train.replace(np.nan, '', regex=True) # replace nan in Province_State with empty string

states = df_train.groupby('Country_Region')['Province_State'].apply(set)

df_train = df_train.groupby(['Country_Region', 'Date']).sum().reset_index() # merge islands



df_pops = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

pops = dict(zip(df_pops['Country (or dependency)'], df_pops['Population (2020)']))

pops['US'] = pops['United States']

pops['Korea, South'] = pops['South Korea']
current_confirmed = df_train.groupby('Country_Region').max().sort_values('ConfirmedCases', ascending=False)

current_fatalities = df_train.groupby('Country_Region').max().sort_values('Fatalities', ascending=False)

current_confirmed_poprate = current_confirmed.loc[:,'ConfirmedCases'] / df_pops.set_index('Country (or dependency)')['Population (2020)']

tops = current_confirmed[:15]

tops = pd.concat([tops, current_confirmed[current_confirmed.index == 'Japan']]) # my country...
df_train['ConfirmedCasesDiff'] = df_train.groupby('Country_Region').diff()['ConfirmedCases']

df_train['FatalitiesDiff'] = df_train.groupby('Country_Region').diff()['Fatalities']
# calculating moving average of the differences

df_train['ConfirmedCasesDiffAvg'] = df_train.groupby('Country_Region').rolling(5)['ConfirmedCasesDiff'].mean().reset_index(level=0)['ConfirmedCasesDiff']

df_train['FatalitiesDiffAvg'] = df_train.groupby('Country_Region').rolling(5)['FatalitiesDiff'].mean().reset_index(level=0)['FatalitiesDiff']
# set the first confirmed date to each rows

df_train['FirstConfirmedDateCountry'] = df_train.query('ConfirmedCases>0').groupby('Country_Region')['Date'].transform('min')

df_train['FirstFatalityDateCountry'] = df_train.query('Fatalities>0').groupby('Country_Region')['Date'].transform('min')
df_train['DaysSinceFirstConfirmed'] = (df_train['Date'] - df_train['FirstConfirmedDateCountry']).dt.days
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of cases')



c = df_train[['Date', 'ConfirmedCases', 'Fatalities']].groupby('Date').sum()

c['ConfirmedCases'].plot(ax=ax1)

c['ConfirmedCases'].plot(ax=ax2)

c['Fatalities'].plot(ax=ax1)

c['Fatalities'].plot(ax=ax2)

ax1.legend(loc='upper left')

ax2.legend()

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of confirmed cases')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['ConfirmedCases']

    c.plot(label=country, ax=ax1)

    c.plot(label=country, ax=ax2)

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, ax = plt.subplots(figsize=(20, 10))

plt.title('Total confirmed cases of COVID-19 per million people from the first case')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    if country not in pops.keys():

        print(f'`{country}` is not found in pops list.', file=sys.stderr)

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['ConfirmedCases']

    c = c / pops[country] * 1e6

    c.plot(label=country, ax=ax)

    ax.annotate(country, xy=(c.index[-1], c.iloc[-1]), size=14)

ax.set_ylabel('Number of cases')

ax.grid()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of confirmed cases per day')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['ConfirmedCasesDiff']

    c.plot(label=country, ax=ax1)

    c[c > 0].plot(label=country, ax=ax2)

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of confirmed cases per day (smoothed)')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['ConfirmedCasesDiffAvg']

    c.plot(label=country, ax=ax1)

    c[c > 0].plot(label=country, ax=ax2)

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of deceased cases')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['Fatalities']

    c.plot(label=country, ax=ax1)

    c[c > 0].plot(label=country, ax=ax2) # for logarithmic scale

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, ax = plt.subplots(figsize=(20, 10))

plt.title('Total deceased cases of COVID-19 per million people from the first confirmed case')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    if country not in pops.keys():

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['Fatalities']

    c = c / pops[country] * 1e6

    c.plot(label=country, ax=ax)

    ax.annotate(country, xy=(c.index[-1], c.iloc[-1]), size=14)

ax.set_xlabel('Days')

ax.set_ylabel('Number of cases')

ax.grid()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of deceased cases per day')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['FatalitiesDiff']

    c.plot(label=country, ax=ax1)

    c[c > 0].plot(label=country, ax=ax2) # for logarithmic scale

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

fig.suptitle('The number of deceased cases per day (smoothed)')

for country, df in df_train.groupby('Country_Region'):

    if country not in tops.index:

        continue

    c = df.query('DaysSinceFirstConfirmed>0').set_index('DaysSinceFirstConfirmed').sort_index()['FatalitiesDiffAvg']

    c.plot(label=country, ax=ax1)

    c[c > 0].plot(label=country, ax=ax2) # for logarithmic scale

    ax1.annotate(country, xy=(c.index[-1], c.iloc[-1]))

    ax2.annotate(country, xy=(c.index[-1], c.iloc[-1]))

ax1.set_ylabel('Number of cases')

ax2.set_yscale('log')
fig, ax = plt.subplots()

c = df_train[(df_train['DaysSinceFirstConfirmed'] > 0) & (df_train['Country_Region'] == 'Italy')].set_index('DaysSinceFirstConfirmed').sort_index()

c['FatalitiesDiff'].plot(label='fatalities diff', ax=ax)

c['ConfirmedCasesDiff'].plot(label='confirmed diff', ax=ax)

ax.legend()
fig, ax = plt.subplots()

c = df_train[(df_train['DaysSinceFirstConfirmed'] > 0) & (df_train['Country_Region'] == 'Italy')].set_index('DaysSinceFirstConfirmed').sort_index()

c['FatalitiesDiffAvg'].plot(label='fatalities diff', ax=ax)

c['ConfirmedCasesDiffAvg'].plot(label='confirmed diff', ax=ax)

ax.legend()