import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
confirmed = pd.read_csv('../input/covid-dataset/time_series_covid19_confirmed_global.csv')

deaths = pd.read_csv('../input/covid-dataset/time_series_covid19_deaths_global.csv')

recovered = pd.read_csv('../input/covid-dataset/time_series_covid19_recovered_global.csv')
confirmed.head()
world_confirmed = confirmed.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)

world_confirmed.head()
dates = list(world_confirmed.columns)
world_confirmed_list = []

for col in world_confirmed:

    world_confirmed_list.append(sum(world_confirmed[col]))
dates_final, world_confirmed_finallist = [], []

for i in range(len(dates)):

    if i%5==0:

        dates_final.append(dates[i])

        world_confirmed_finallist.append(world_confirmed_list[i])
plt.style.use('fivethirtyeight')

plt.figure(figsize=(10, 6))

plt.plot(dates_final, world_confirmed_finallist, 'o-b')

plt.xticks(fontsize=10, rotation=90)

plt.xlabel('Dates')

plt.ylabel('No. of Confirmed Cases')

plt.title('World Confirmed Cases Analysis')

plt.grid(True)

plt.tight_layout()

plt.show()
deaths.head()
world_deaths = deaths.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)

world_deaths.head()
dates = list(world_deaths.columns)
world_deaths_list = []

for col in world_deaths:

    world_deaths_list.append(sum(world_deaths[col]))
dates_final, world_deaths_finallist = [], []

for i in range(len(dates)):

    if i%5==0:

        dates_final.append(dates[i])

        world_deaths_finallist.append(world_deaths_list[i])
plt.style.use('fivethirtyeight')

plt.figure(figsize=(10, 6))

plt.plot(dates_final, world_deaths_finallist, 'o-b')

plt.xticks(fontsize=10, rotation=90)

plt.xlabel('Dates')

plt.ylabel('No. of Deaths')

plt.title('Death Analysis')

plt.grid(True)

plt.tight_layout()

plt.show()
recovered.head()
world_recovered = recovered.drop(labels=['Province/State', 'Lat', 'Long', 'Country/Region'], axis=1)

world_recovered.head()
dates = list(world_recovered.columns)
world_recovered_list = []

for col in world_recovered:

    world_recovered_list.append(sum(world_recovered[col]))
dates_final, world_recovered_finallist = [], []

for i in range(len(dates)):

    if i%5==0:

        dates_final.append(dates[i])

        world_recovered_finallist.append(world_recovered_list[i])
plt.style.use('fivethirtyeight')

plt.figure(figsize=(10, 5))

plt.plot(dates_final, world_recovered_finallist, 'o-b')

plt.xticks(fontsize=10, rotation=90)

plt.xlabel('Dates')

plt.ylabel('No. of Recoveries')

plt.title('Recovery Analysis')

plt.grid(True)

plt.tight_layout()

plt.show()
confirmed.head()
countries_confirmed = confirmed.drop(labels=['Province/State', 'Lat', 'Long'], axis=1)

countries_confirmed.head()
countries_confirmed['Total'] = countries_confirmed['1/22/20']

for col in countries_confirmed.columns:

    if col == 'Country/Region' or col == '1/22/20':

        continue

    countries_confirmed['Total'] += countries_confirmed[col]
countries_confirmed.head()
countries_confirmed = countries_confirmed.groupby(['Country/Region']).sum()

countries_confirmed.reset_index(inplace=True)

countries_confirmed_1 = countries_confirmed[['Country/Region', 'Total']]

countries_confirmed_1 = countries_confirmed[countries_confirmed['Total'] > 5000000]

countries_confirmed_1.head()
countries = list(countries_confirmed_1['Country/Region'])

confirmed_list = list(countries_confirmed_1.Total)

print(countries)

print(confirmed_list)
plt.style.use('fivethirtyeight')

y_pos = np.arange(len(countries))

plt.figure(figsize=(10, 6))

plt.barh(y_pos, confirmed_list, align='center')

plt.yticks(ticks=y_pos, labels=countries, fontsize=9)

plt.xlabel('No. of confirmed cases')

plt.ylabel('Countries')

plt.gca().invert_yaxis()

plt.title('Confirmed cases')

plt.grid(True)

plt.show()
deaths.head()
countries_deaths = deaths.drop(labels=['Province/State', 'Lat', 'Long'], axis=1)

countries_deaths.head()
countries_deaths['Total'] = countries_deaths['1/22/20']

for col in countries_deaths.columns:

    if col == 'Country/Region' or col == '1/22/20':

        continue

    countries_deaths['Total'] += countries_deaths[col]
countries_deaths.head()
countries_deaths = countries_deaths.groupby(['Country/Region']).sum()

countries_deaths.reset_index(inplace=True)

countries_deaths_1 = countries_deaths[['Country/Region', 'Total']]

countries_deaths_1 = countries_deaths[countries_deaths['Total'] > 50000]

countries_deaths_1.head()
countries = list(countries_deaths_1['Country/Region'])

death_list = list(countries_deaths_1.Total)

print(countries)

print(death_list)
plt.style.use('fivethirtyeight')

y_pos = np.arange(len(countries))

plt.figure(figsize=(10, 8))

plt.barh(y_pos, death_list, align='center')

plt.yticks(ticks=y_pos, labels=countries, fontsize=8)

plt.xlabel('No. of death cases')

plt.ylabel('Countries')

plt.gca().invert_yaxis()

plt.title('Death cases')

plt.grid(True)

plt.show()
recovered.head()
countries_recovered = recovered.drop(labels=['Province/State', 'Lat', 'Long'], axis=1)

countries_recovered.head()
countries_recovered['Total'] = countries_recovered['1/22/20']

for col in countries_recovered.columns:

    if col == 'Country/Region' or col == '1/22/20':

        continue

    countries_recovered['Total'] += countries_recovered[col]
countries_recovered.head()
countries_recovered = countries_recovered.groupby(['Country/Region']).sum()

countries_recovered.reset_index(inplace=True)

countries_recovered_1 = countries_recovered[['Country/Region', 'Total']]

countries_recovered_1 = countries_recovered[countries_recovered['Total'] > 5000000]

countries_recovered_1.head()
countries = list(countries_recovered_1['Country/Region'])

recovered_list = list(countries_recovered_1.Total)

print(countries)

print(recovered_list)
plt.style.use('fivethirtyeight')

y_pos = np.arange(len(countries))

plt.figure(figsize=(10, 6))

plt.barh(y_pos, recovered_list, align='center')

plt.yticks(ticks=y_pos, labels=countries, fontsize=10)

plt.xlabel('No. of recovered cases')

plt.ylabel('Countries')

plt.gca().invert_yaxis()

plt.title('Recoveries cases')

plt.grid(True)

plt.show()
world_data = pd.merge(countries_confirmed, countries_deaths, on='Country/Region')

world_data = pd.merge(world_data, countries_recovered, on='Country/Region')

world_data = world_data[['Country/Region', 'Total_x', 'Total_y', 'Total']]

world_data = world_data.rename(columns={'Total_x':'Confirmed', 'Total_y':'Deaths', 'Total':'Recovered'})
world_data = world_data.set_index('Country/Region')

world_data
def print_pie(country, world_data=world_data):

    slices = [int(world_data.loc[[country]].Confirmed),

                  int(world_data.loc[[country]].Deaths),

                  int(world_data.loc[[country]].Recovered)]

    labels = ['Confirmed', 'Deaths', 'Recovered']

    colors = ['#2EC6DE', '#C70D26', '#6EB323']

    plt.pie(slices, labels=labels, colors=colors, autopct='%1.1f%%',

            wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    plt.title(country+' Records')

    plt.grid(True)

    plt.tight_layout()

    plt.show()
print_pie(input('Enter a country(first letter caps): '))
print_pie(input('Enter a country(first letter caps): '))
print_pie(input('Enter a country(first letter caps): '))