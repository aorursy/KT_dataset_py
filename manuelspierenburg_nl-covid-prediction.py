# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['ObservationDate', 'Last Update'])
# cleanup data

df = df.rename(columns={'Country/Region': 'Country', 'Province/State': 'Region', 'Last Update': 'Updated', 'ObservationDate': 'Date'})
df['New'] = df.sort_values('Date').groupby('Country').Confirmed.diff()
# overall

total = df.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum()

total.plot()
# countries with most cases

top = df.groupby('Country')['Confirmed'].sum().sort_values(ascending=False).head(10)
# plot trend for top countries

for c in top.index:

    df30 = df.loc[df.Country == c]

    df30.groupby('Date')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index(drop=True).plot(title=c)

    plt.show()
countries = ['Italy', 'Switzerland', 'Netherlands']

df70 = df.loc[df.Country.isin(countries)]

df71 = df70.groupby(['Date', 'Country'])[['Confirmed', 'Deaths', 'Recovered', 'New']].sum()

df72 = df71.loc[df71.Confirmed >= 10]
# compare italy, switzerland and netherlands

# day one is first day with 10 or more confirmed cases

ax = df72.reset_index().groupby(['Country']).apply(lambda gr: gr.reset_index(drop=True)).unstack(level=0).Confirmed.plot()

ax.set_yscale('log')
# prediction for netherlands

df_nl = df.loc[df.Country == 'Netherlands']

dfnl1 = df_nl.groupby(['Date'])[['Confirmed', 'Deaths', 'Recovered', 'New']].sum()
# get current death rate for prediction

last_row = dfnl1.tail(1)

current_death_rate = (last_row.Deaths / last_row.Confirmed)[0]

print(f'current death rate: {current_death_rate:.4f}')
# get average death rate

dfnl2 = dfnl1.loc[dfnl1.Deaths > 0]

mean_death_rate = (dfnl2.Deaths / dfnl2.Confirmed).mean()

print(f'mean death rate: {mean_death_rate:.4f}')
# calculate average growth

nl_growth = dfnl1.Confirmed.diff().fillna(0).pct_change() + 1

nl_growth.replace([np.inf, -np.inf], np.nan, inplace=True)

dfnl1['Growth'] = nl_growth

mean_growth = nl_growth.dropna().mean()

print(f'mean_growth: {mean_growth:.4f}')
dfnl3 = dfnl1.copy()

days = 31

for d in range(days):

    last_row = dfnl3.tail(1)

    new_date = last_row.index + pd.Timedelta(1, 'D')

    new_confirmed = int(last_row.Confirmed[0] * mean_growth)

    new_deaths = int(new_confirmed * current_death_rate)

    new_row = pd.DataFrame(index=new_date, data= {'Confirmed': new_confirmed, 'Deaths': new_deaths})

    dfnl3 = dfnl3.append(new_row, sort=False)
# predict number of recovered people

recovery_days = 14 # how many days until somebody is recoverd

dfnl3 = dfnl3.sort_values('Date')

dfnl3['New'] = dfnl3.Confirmed.diff()

dfnl3['Predicted_Recovered'] = dfnl3.New.shift(recovery_days).fillna(0)

#dfnl3['Predicted_Recovered'] = dfnl3['Predicted_Recovered'] - dfnl3['Deaths']
ax = dfnl3[['Confirmed', 'Predicted_Recovered', 'Deaths']].plot()

today = pd.Timestamp.today()

ax.axvline(today, c='r')

ax.set_yscale('log')
dfnl3.tail()