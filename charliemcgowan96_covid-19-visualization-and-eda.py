import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

import math



register_matplotlib_converters()

sns.set()



from pathlib import Path

data_dir = Path('../input/covid19-global-forecasting-week-1')
df = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])



# Renaming, re-ordering and creating a current cases column

df = df.rename(columns={'Deaths':'Deceased', 'Province/State':'Province', 'Country/Region':'Country'})

df['Current'] = df['Confirmed'] - df['Deceased'] - df['Recovered']

df = df[['Country','Province', 'Lat', 'Long', 'Date','Confirmed', 'Current', 'Recovered', 'Deceased']]

df.head()
# Grouping the data globally

globally = df[df.columns.values[-5:]].groupby('Date').sum().reset_index()



# Creating a loop for the various plots

plt.subplots(nrows=2, ncols=2, figsize=(20,10))

i = 0

for graph in globally.columns.values[1:]:

    i += 1

    plt.subplot(2,2,i)

    plt.plot(globally[graph]/1e3, label = graph)

    plt.title('Global ' + graph + ' Cases')

    plt.xlabel('Days (starting 2020-01-22)')

    plt.ylabel('Thousand Cases')

    plt.xlim(globally.index.min(), globally.index.max())

    plt.legend(loc='best')



print('As of', globally.Date.dt.date.max(),':')

print(int(globally.Confirmed.max()), 'Confirmed cases')

print(int(globally.Current.max()), 'Current cases')

print(int(globally.Recovered.max()), 'Recovered cases')

print(int(globally.Deceased.max()), 'Deceased cases')
resolved_cases = globally['Recovered'].sum() + globally['Deceased'].sum()

mortality = round(100 * (globally['Deceased'].sum() / resolved_cases), 2)

print('Currently the global mortality rate is:', mortality, '%')
latest = df.loc[df['Date'] == df['Date'].max()].groupby('Country').sum().reset_index()



latest = latest.sort_values(by=['Confirmed'], ascending=False).reset_index(drop=True)

top_10 = latest.loc[:9]

top_10_bar = top_10.set_index('Country')[top_10.columns[3:]]

top_10_names = top_10['Country']



(top_10_bar/1e3).plot.bar(figsize=(20,5))

plt.ylabel('Thousand Cases')
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING

import plotly.express as px



fig = px.choropleth(latest, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Confirmed", range_color=[1,50000], 

                    color_continuous_scale='Reds', 

                    title='Global view of Confirmed Cases')

fig.show()
europe = list(['Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Ireland',

               'Italy', 'Latvia','Luxembourg','Lithuania','Malta','Norway','Netherlands','Poland','Portugal','Romania','Slovakia','Slovenia',

               'Spain', 'Sweden', 'United Kingdom', 'Iceland', 'Russia', 'Switzerland', 'Serbia', 'Ukraine', 'Belarus',

               'Albania', 'Bosnia and Herzegovina', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia'])



european_countries = df[df['Country'].isin(europe)]

latest_european = european_countries.loc[european_countries['Date'] == european_countries['Date'].max()].groupby('Country').sum().reset_index()



China = df.loc[df['Country'] == 'China'].groupby('Date').sum().reset_index()

Iran = df.loc[df['Country'] == 'Iran'].groupby('Date').sum().reset_index()

USA = df.loc[df['Country'] == 'US'].groupby('Date').sum().reset_index()

EU = european_countries.groupby('Date').sum().reset_index()



# Creating a loop for the various plots

plt.subplots(nrows=2, ncols=2, figsize=(20,10))



plt.subplot(2,2,1)

plt.plot(China['Current']/1e3, label='Current', color='green')

plt.plot(China['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(China['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'China' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(globally.index.min(), globally.index.max())

plt.legend(loc='best')



plt.subplot(2,2,2)

plt.plot(EU['Current']/1e3, label='Current', color='green')

plt.plot(EU['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(EU['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'EU' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(globally.index.min(), globally.index.max())

plt.legend(loc='best')



plt.subplot(2,2,3)

plt.plot(Iran['Current']/1e3, label='Current', color='green')

plt.plot(Iran['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(Iran['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'Iran' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(globally.index.min(), globally.index.max())

plt.legend(loc='best')



plt.subplot(2,2,4)

plt.plot(USA['Current']/1e3, label='Current', color='green')

plt.plot(USA['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(USA['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'USA' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(globally.index.min(), globally.index.max())

plt.legend(loc='best')
fig = px.choropleth(latest_european, locations="Country", 

                    locationmode='country names', color="Confirmed", 

                    hover_name="Confirmed", range_color=[1,5000], 

                    color_continuous_scale='Reds', 

                    title='European view of Current Cases', scope='europe')#, height=800, width= 1400)

# fig.update(layout_coloraxis_showscale=False)

fig.show()
Italy = df.loc[df['Country'] == 'Italy'].groupby('Date').sum().reset_index()

UK = df.loc[df['Country'] == 'United Kingdom'].groupby('Date').sum().reset_index()

France = df.loc[df['Country'] == 'France'].groupby('Date').sum().reset_index()

Spain = df.loc[df['Country'] == 'Spain'].groupby('Date').sum().reset_index()

Germany = df.loc[df['Country'] == 'Germany'].groupby('Date').sum().reset_index()

Switzerland = df.loc[df['Country'] == 'Switzerland'].groupby('Date').sum().reset_index()



# Creating a loop for the various plots

plt.subplots(nrows=3, ncols=2, figsize=(20,10))



plt.subplot(2,3,1)

plt.plot(Italy['Current']/1e3, label='Current', color='green')

plt.plot(Italy['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(Italy['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'Italy' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')



plt.subplot(2,3,2)

plt.plot(Germany['Current']/1e3, label='Current', color='green')

plt.plot(Germany['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(Germany['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'Germany' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')



plt.subplot(2,3,3)

plt.plot(Spain['Current']/1e3, label='Current', color='green')

plt.plot(Spain['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(Spain['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'Spain' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')



plt.subplot(2,3,4)

plt.plot(France['Current']/1e3, label='Current', color='green')

plt.plot(France['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(France['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'France' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')



plt.subplot(2,3,5)

plt.plot(Switzerland['Current']/1e3, label='Current', color='green')

plt.plot(Switzerland['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(Switzerland['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'Switzerland' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')



plt.subplot(2,3,6)

plt.plot(UK['Current']/1e3, label='Current', color='green')

plt.plot(UK['Recovered']/1e3, label='Recovered', color='orange')

plt.plot(UK['Deceased']/1e3, label='Deceased', color='red')

plt.title('Global ' + 'UK' + ' Cases')

plt.xlabel('Days (starting 2020-01-22)')

plt.ylabel('Thousand Cases')

plt.xlim(30, 60)

plt.legend(loc='best')