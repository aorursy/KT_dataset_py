import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams["figure.figsize"] = [15, 5]
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
p = df.groupby(['ObservationDate']).sum().reset_index().set_index('ObservationDate')

fig, ax = plt.subplots()

ax.plot(p.index, p['Confirmed'] - p['Deaths'] - p['Recovered'], 'b', linewidth=3, label='Active')

ax.plot(p.index, p['Deaths'], 'r+', label='Deaths')

ax.plot(p.index, p['Recovered'], 'g--', label='Recovered')

ax.plot(p.index, p['Confirmed'], 'orange', label='Confirmed')

plt.xticks(rotation=45)

ax.legend()

ax.grid()

plt.title('World')

ax.xaxis.set_major_locator(plt.MaxNLocator(30))

plt.show()
countries = {'Mainland China':None, 'Singapore': None, 'Italy':None, 'Spain':None, 'US':None, 'Germany': None, 'Poland': None, 'Sweden':None, 'Belarus': None, 'Russia':None}

for country in countries.keys():

    countries[country] = df[df['Country/Region'] == country].groupby(['ObservationDate', 'Country/Region']).sum().reset_index().set_index('ObservationDate')

    countries[country] = countries[country].drop(countries[country][countries[country]['Confirmed'] < 50].index)
for country, p in countries. items():

    fig, ax = plt.subplots()

    ax.plot(p.index, p['Confirmed'] - p['Deaths'] - p['Recovered'], 'b', linewidth=3, label='Active')

    ax.plot(p.index, p['Deaths'], 'r+', label='Deaths')

    ax.plot(p.index, p['Recovered'], 'g--', label='Recovered')

    ax.plot(p.index, p['Confirmed'], 'orange', label='Confirmed')

    plt.xticks(rotation=45)

    ax.legend()

    ax.grid()

    plt.title(country)

    ax.xaxis.set_major_locator(plt.MaxNLocator(30))

    plt.show()