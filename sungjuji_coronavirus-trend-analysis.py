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
import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
#data_0225 contains latest count of confirmed/deaths/recovered count of Corona virus for each province/state

data_0225=pd.read_csv("../input/corona19/0225.csv")

data_0225.head()
#data_total contains total confirmed/deaths/recovered cases for Corona virus for each day from its first outbreak date to Feb. 25

#this is what original data looks like

data_total=pd.read_csv("/kaggle/input/corona19/corona_compiled.csv")

data_total.head()
#showing current data type for each column

data_total.info()
#convert NaN values to "0"

df = data_total.fillna(0)



#convert strings to date type for necessary columns

df[['Date']] = pd.to_datetime(df['Date'])

df[['Last Update']] = pd.to_datetime(df['Last Update'])



df.head()
#here is an world's accumulative daily status table for Coronavirus outbreak

daily_cases = df.groupby("Date", as_index=False)["Confirmed", "Deaths", "Recovered"].sum().set_index("Date")

daily_cases
#Following table shows world total coronavirus cases as of 2/25/2020

confirmed = data_0225['Confirmed'].sum()

death = data_0225['Deaths'].sum()

recovered = data_0225['Recovered'].sum()



df = pd.DataFrame({"World Total": [confirmed, death, recovered]})

df.rename(index = {0: "Confirmed", 1: "Death", 2: "Recovered"}).T
#Line graph displaying the trend for confirmed, deaths, recovered cases of Coronavirus worldwide.

daily_cases.plot(kind='line', figsize=(28,10))

plt.title('Accumulated COVID-19 Case Trend Over Time', size = 40)

#Table that shows accumulated Coronavirus status per each country as of 2/25/2020. It is sorted by most to least confirmed cases.



country_cases = data_0225.groupby("Country/Region", as_index=False)["Confirmed", "Deaths", "Recovered"].sum().set_index("Country/Region")

country_cases.sort_values(by=['Confirmed'], ascending=False)
#Bar graph displaying the trend for confirmed, deaths, recovered cases of Coronavirus worldwide per each country.

country_cases.plot(kind='bar', figsize=(28,10))

plt.title('Total COVID-19 Cases per Country', size = 40)
#Because China outnumbers other countries, I've decided to exclude China from the dataset to have clearer visualization for other countries

nonchina_cases = country_cases.drop('Mainland China', axis =0)

nonchina_cases.sort_values(by='Confirmed', ascending = False)
#bar graph displaying the accumulated confirmed cases for Coronavirus per country withouth China

nonchina_confirmed = nonchina_cases.sort_values(by=['Confirmed'], ascending = False).Confirmed

nonchina_confirmed.plot(kind='bar', figsize=(28,10), color = "SlateGray")

plt.title('Number of Confirmed Cases', size = 40)
nonchina.deaths = nonchina_cases.sort_values(by=['Deaths'], ascending = False).Deaths

nonchina.deaths.plot(kind='bar', figsize=(28,10), color = 'DarkRed')

plt.title('Number of Death Cases', size = 40)
nonchina.recovered = nonchina_cases.sort_values(by=['Recovered'], ascending = False).Recovered

nonchina.recovered.plot(kind='bar', figsize=(28,10), color = "DarkGreen")

plt.title('Number of Recovered Cases', size = 40)