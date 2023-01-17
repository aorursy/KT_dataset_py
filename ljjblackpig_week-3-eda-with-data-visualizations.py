import pandas as pd

import random

import numpy as np



import matplotlib.pyplot as plt
df = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')
df['month'] = pd.DatetimeIndex(df['Date']).month

df.head()
sorted_by_country = df[df.Date == '2020-04-07'].groupby('Country_Region').agg('sum').reset_index().sort_values(

    by = ['ConfirmedCases', 'Fatalities'], ascending = False)

sorted_by_country['Death Ratio'] = sorted_by_country.Fatalities / sorted_by_country.ConfirmedCases * 100

sorted_by_country.drop(columns = ['month'], inplace = True)

sorted_by_country.head(20)
march = df[df.Date == '2020-03-31'].groupby('Country_Region').agg('sum').reset_index()

april = df[df.Date == '2020-04-07'].groupby('Country_Region').agg('sum').reset_index()

Confirmed = april.ConfirmedCases - march.ConfirmedCases

Fatalities = april.Fatalities - march.Fatalities

april['ConfirmedCases_new'] = Confirmed

april['Fatalities_new'] = Fatalities
april_data = april.sort_values(by = ['ConfirmedCases_new', 'Fatalities_new'], ascending = False)

april_data['Death Ratio'] = april_data.Fatalities_new / april_data.ConfirmedCases_new * 100

april_data.drop(columns = ['month'], inplace = True)

april_data.head(20)
fig = plt.figure(figsize = (15, 8))



for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 9)):

    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()

    temp['month'] = pd.DatetimeIndex(temp['Date']).month

    temp = temp.groupby('month').agg('sum').reset_index()

    name = 'ax' + str(idx)

    name = fig.add_subplot(3, 3, idx+1)

    name.bar(temp.month, temp.ConfirmedCases)

    name.set_title("Montly Confirmed Cases in {}".format(country))

    name.set_xticks([1,2,3,4])

    name.grid(True)

            

plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (15, 8))



for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 9)):

    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()

    temp['month'] = pd.DatetimeIndex(temp['Date']).month

    temp = temp.groupby('month').agg('sum').reset_index()

    name = 'ax' + str(idx)

    name = fig.add_subplot(3, 3, idx+1)

    name.bar(temp.month, temp.Fatalities)

    name.set_title("Montly Death in {}".format(country))

    name.set_xticks([1,2,3,4])

    name.grid(True)

            

plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (15, 8))

every_nth = 10



for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 10)):

    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()

    name = 'ax' + str(idx)

    name = fig.add_subplot(5, 2, idx+1)

    name.plot(temp.Date, temp.ConfirmedCases)

    name.plot(temp.Date, temp.Fatalities)

    name.set_title("Day by Day Confirmed Cases vs Fatalities in {}".format(country))

    name.legend(('Confirmed Cases', 'Fatalities'))

    name.grid(True)

    for n, label in enumerate(name.xaxis.get_ticklabels()):

        if n % every_nth != 0:

            label.set_visible(False)

            

plt.tight_layout()

plt.show()
fig = plt.figure(figsize = (15, 8))

every_nth = 10



for idx, country in enumerate(random.sample(list(df.Country_Region.unique()), 10)):

    temp = df[df['Country_Region'] == country].groupby('Date').agg('sum').reset_index()

    temp['Death_Ratio'] = temp.Fatalities / temp.ConfirmedCases * 100

    name = 'ax' + str(idx)

    name = fig.add_subplot(5, 2, idx+1)

    name.plot(temp.Date, temp.Death_Ratio)

    name.axhline(temp.Death_Ratio.mean(), linestyle = '--', color = 'red')

    name.set_title("Day by Day Death Ratio (%) in {}".format(country))

    name.legend(['Death Ratio', 'Average Death Ratio'], fontsize = 8)

    name.grid(True)

    for n, label in enumerate(name.xaxis.get_ticklabels()):

        if n % every_nth != 0:

            label.set_visible(False)

            

plt.tight_layout()

plt.show()