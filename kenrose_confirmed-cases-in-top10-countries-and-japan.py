# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import datetime as dt

dt_string = dt.datetime.now(dt.timezone.utc).strftime("%d/%m/%Y %H:%M:%S %z")

print(f"Kernel last updated: {dt_string}")
cases = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

cases.head()
# create dayly cases data by country

grp = cases.groupby(['ObservationDate', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

grp = grp.reset_index()

grp.head()
grp.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

grp['Date'] = pd.to_datetime(grp['Date'], format='%m/%d/%Y')



grp['Confirmed'] = grp['Confirmed'].astype(np.int64)

grp['Deaths'] = grp['Deaths'].astype(np.int64)

grp['Recovered'] = grp['Recovered'].astype(np.int64)



grp['Active'] = grp['Confirmed'] - grp['Recovered'] - grp['Deaths']

grp.head()
# pick up Japanese data

df_japan = grp[grp['Country'] == 'Japan']

df_japan.reset_index(inplace=True, drop=True)

df_japan.head()
# plot Confirmed in Japan

fig, ax = plt.subplots(figsize=(15, 8))

plt.plot(df_japan['Date'], df_japan['Confirmed'])

ax.grid(which='major', axis='y', linestyle='-')

ax.grid(which='major', axis='x', linestyle='-')

ax.set_xlabel("Date")

ax.set_ylabel("Confirmed")

ax.set_title("Confirmed COVID-19 cases in Japan")

plt.show()
# get data of since confirmed count was over 100

df_japan = df_japan[df_japan['Confirmed']>=100]

df_japan.reset_index(inplace=True, drop=True)

df_japan.head()
# plot number of Confirmed in Japan

fig, ax = plt.subplots(figsize=(15, 8))

plt.plot(df_japan.index, df_japan['Confirmed'], label=f'Confirmed ({df_japan["Confirmed"].max()})')



ax.set_yscale('log')  # axis y 'log' scale

ax.grid(which='major', axis='y', linestyle='-')

ax.grid(which='major', axis='x', linestyle='-')



ax.set_xlabel("Number of days since confirmed count was over 100")

ax.set_ylabel("Confirmed")

ax.set_title("Confirmed COVID-19 cases in Japan")



plt.legend(loc='best')

plt.show()
# get data of last date (sorted by 'Confirmed')

df_lastDate = grp[grp['Date']==grp['Date'].max()].sort_values('Confirmed', ascending=False)

df_lastDate.head(10)
# TOP10 countries

top10Countries = df_lastDate['Country'][:10].values

top10Countries
# Plor chart (since confirmed count was over 100)

def PlotCountryChart(country, color=None, lineStyle=None):

    df_country = grp[grp['Country'] == country]

    df_country = df_country[df_country['Confirmed']>=100]

    df_country.reset_index(inplace=True, drop=True)

    plt.plot(df_country.index, df_country['Confirmed'], 

             color=color, linestyle=lineStyle, 

             label=f'{country} ({df_country["Confirmed"].max()})')

# japan declares state of emergency

df_jpn = grp[grp['Country'] == 'Japan'].reset_index(drop=True)

df_jpn = df_jpn[df_jpn['Confirmed']>=100]

df_jpn.reset_index(inplace=True, drop=True)



df_jpn['Point'] = 0

df_jpn.loc[df_jpn['Date']=='2020-04-07', 'Point'] = df_jpn.loc[df_jpn['Date']=='2020-04-07']['Confirmed']  # date of declares state of emergency

df_jpn.loc[df_jpn['Date']=='2020-04-21', 'Point'] = df_jpn.loc[df_jpn['Date']=='2020-04-21']['Confirmed']  # after 2 weeks

df_jpn.tail()
# draw ’Confirmed’ count in top10 countries and Japan 

fig, ax = plt.subplots(figsize=(20, 10))

for country in top10Countries:

    PlotCountryChart(country)



# add japan data

country = 'Japan'

PlotCountryChart('Japan', 'r', '--')



plt.plot(df_jpn.index, df_jpn['Point'], 

         color='pink', linestyle=':', 

         label='japan declares state of emergency and after 2weeks')





# draw line of cases double every n days

plt.plot([0, 140], [100, 100 * 2**(140/15)], 'b:')

plt.plot([0, 140], [100, 100 * 2**(140/10)], 'b:')

plt.plot([0, 115], [100, 100 * 2**(115/7)], 'g:')

plt.plot([0, 83], [100, 100 * 2**(83/5)], 'y:')

plt.plot([0, 50], [100, 100 * 2**(50/3)], 'r:')

plt.plot([0, 33], [100, 100 * 2**(33/2)], 'r:')



fig.text(0.8, 0.55, 'cases double every 15 days', size = 10, color = "b")

fig.text(0.8, 0.75, 'cases double every 10 days', size = 10, color = "b")

fig.text(0.7, 0.85, 'cases double every 7 days', size = 10, color = "g")

fig.text(0.5, 0.85, 'cases double every 5 days', size = 10, color = "y")

fig.text(0.35, 0.85, 'cases double every 3 days', size = 10, color = "r")

fig.text(0.25, 0.85, 'cases double every 2 days', size = 10, color = "r")





ax.set_yscale('log')  # axis y 'log' scale

ax.grid(which='major', axis='y', linestyle='-')

ax.grid(which='major', axis='x', linestyle='-')



ax.set_xlabel("Number of days since confirmed count over 100")

ax.set_ylabel("Confirmed")

ax.set_title("Confirmed COVID-19 cases in top10 countries and Japan")



plt.legend(loc='best')

plt.show()