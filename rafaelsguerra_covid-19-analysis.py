# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

covid_19 = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
covid_19.rename(columns={'Country/Region': 'Country'}, inplace=True)

confirmed.rename(columns={'Country/Region': 'Country'}, inplace=True)

deaths.rename(columns={'Country/Region': 'Country'}, inplace=True)

recovered.rename(columns={'Country/Region': 'Country'}, inplace=True)
covid_19.head()
confirmed.head()
covid_19.query('Country == "Brazil"').groupby('Last Update')[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
import matplotlib.pyplot as plt
covid_19_confirmed = covid_19.groupby('ObservationDate')['Confirmed'].sum().reset_index()

covid_19_deaths = covid_19.groupby('ObservationDate')['Deaths'].sum().reset_index()

covid_19_recovered = covid_19.groupby('ObservationDate')['Recovered'].sum().reset_index()
covid_19_confirmed.head()
fig = plt.figure(figsize=(14, 5))

axes= fig.add_axes([0.1,0.1,0.8,0.8])





axes.plot(covid_19_confirmed['ObservationDate'], covid_19_confirmed['Confirmed'])

axes.plot(covid_19_deaths['ObservationDate'], covid_19_deaths['Deaths'], color='r')

axes.plot(covid_19_recovered['ObservationDate'], covid_19_recovered['Recovered'], color='g')



plt.xticks(rotation=-90)



plt.title('COVID-19 Worldwide cases until march 16th')

plt.xlabel('Date')

plt.ylabel('Cases')

plt.legend()



plt.show()
brazil_confirmed = covid_19.query("Country == 'Brazil'").groupby('ObservationDate')['Confirmed'].sum().reset_index()

brazil_deaths = covid_19.query("Country == 'Brazil'").groupby('ObservationDate')['Deaths'].sum().reset_index()

brazil_recovered = covid_19.query("Country == 'Brazil'").groupby('ObservationDate')['Recovered'].sum().reset_index()
plt.plot(brazil_confirmed['ObservationDate'], brazil_confirmed['Confirmed'])

plt.plot(brazil_deaths['ObservationDate'], brazil_deaths['Deaths'], color='r')

plt.plot(brazil_recovered['ObservationDate'], brazil_recovered['Recovered'], color='g')



plt.title('COVID-19 Cases in Brazil until march 16th')

plt.xlabel('Date')

plt.ylabel('Cases')



plt.legend()

plt.xticks(rotation=-90)



plt.show()
china_confirmed = covid_19.query("Country == 'Mainland China'").groupby('ObservationDate')['Confirmed'].sum().reset_index()

china_deaths = covid_19.query("Country == 'Mainland China'").groupby('ObservationDate')['Deaths'].sum().reset_index()

china_recovered = covid_19.query("Country == 'Mainland China'").groupby('ObservationDate')['Recovered'].sum().reset_index()



fig = plt.figure(figsize=(14, 5))

axes= fig.add_axes([0.1,0.1,0.8,0.8])



plt.plot(china_confirmed['ObservationDate'], china_confirmed['Confirmed'])

plt.plot(china_deaths['ObservationDate'], china_deaths['Deaths'], color = 'r')

plt.plot(china_recovered['ObservationDate'], china_recovered['Recovered'], color = 'g')



plt.title('COVID-19 cases in China until march 16th')

plt.xlabel('Date')

plt.ylabel('Cases')



plt.legend()

plt.xticks(rotation=-90)



plt.show()
italy_confirmed = covid_19.query("Country == 'Italy'").groupby('ObservationDate')['Confirmed'].sum().reset_index()

italy_deaths = covid_19.query("Country == 'Italy'").groupby('ObservationDate')['Deaths'].sum().reset_index()

italy_recovered = covid_19.query("Country == 'Italy'").groupby('ObservationDate')['Recovered'].sum().reset_index()



fig = plt.figure(figsize=(14, 5))

axes= fig.add_axes([0.1,0.1,0.8,0.8])



plt.plot(italy_confirmed['ObservationDate'], italy_confirmed['Confirmed'])

plt.plot(italy_deaths['ObservationDate'], italy_deaths['Deaths'], color = 'r')

plt.plot(italy_recovered['ObservationDate'], italy_recovered['Recovered'], color = 'g')



plt.title('COVID-19 cases in Italy until march 16th')

plt.xlabel('Date')

plt.ylabel('Cases')



plt.legend()

plt.xticks(rotation=-90)



plt.show()