# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import date

import math  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path_global = '/kaggle/input/corona-virus-report/covid_19_clean_complete.csv'

path_confirmed = '/kaggle/input/colombia-covid19-complete-dataset/covid-19-colombia-confirmed.csv'

path_colombia = '/kaggle/input/colombia-covid19-complete-dataset/covid-19-colombia.csv'

path_deaths ='/kaggle/input/colombia-covid19-complete-dataset/covid-19-colombia-deaths.csv'

path_all = '/kaggle/input/colombia-covid19-complete-dataset/covid-19-colombia-all.csv'





df_global = pd.read_csv(path_global, parse_dates=['Date'])

df_confirmed = pd.read_csv(path_confirmed)

df_colombia = pd.read_csv(path_colombia, parse_dates=['date'])

#df_deaths = pd.read_csv(path_deaths)

df_all = pd.read_csv(path_all)

countryList = ['Colombia', 'Canada', 'Italy', 'Spain', 'South Korea', 'Mexico']

df_global.head()
# Data for countries of interest

df_Colombia = df_global[df_global['Country/Region'] == 'Colombia']

df_Canada =  df_global[df_global['Country/Region'] == 'Canada']

df_Italy =  df_global[df_global['Country/Region'] == 'Italy']

df_Spain =  df_global[df_global['Country/Region'] == 'Spain']

df_SouthKorea =  df_global[df_global['Country/Region'] == 'South Korea']

df_Mexico =  df_global[df_global['Country/Region'] == 'Mexico']
# Counting days from first confirmed case in each country (normalize)

# In the case of Colombia, data is accumulative 

initialDate_Colombia = df_Colombia[df_Colombia['Confirmed'] >= 1]

initialDate_Colombia = initialDate_Colombia['Date'].min()

print(initialDate_Colombia)



maxDate_Colombia = df_Colombia['Date'].max()

print(maxDate_Colombia)

df_Colombia = df_Colombia[:]

days_Colombia = (df_Colombia['Date'] - initialDate_Colombia)



df_Colombia['Day Number'] = days_Colombia.dt.days

df_Colombia['Confirmed Change'] = df_Colombia['Confirmed'].diff()

df_Colombia.tail()
# Counting days from first confirmed case in each country (normalize)

# For Canada, the data is by province, I group by for this first comparison

initialDate_Canada = df_Canada[df_Canada['Confirmed'] >= 1]

initialDate_Canada = initialDate_Canada['Date'].min()

print(initialDate_Canada)



maxDate_Canada = df_Canada['Date'].max()

print(maxDate_Canada)



df_Canada = df_Canada.groupby(['Date']).sum().reset_index()

df_Canada = df_Canada[:]

df_Canada['Day Number'] = (df_Canada['Date'] - initialDate_Canada).dt.days

df_Canada['Confirmed Change'] = df_Canada['Confirmed'].diff()

df_Canada.tail()
# Counting days from first confirmed case in each country (normalize)

initialDate_Italy = df_Italy[df_Italy['Confirmed'] >= 1]

initialDate_Italy = initialDate_Italy['Date'].min()

print(initialDate_Italy)

df_Italy = df_Italy[:]

df_Italy['Day Number'] = (df_Italy['Date'] - initialDate_Italy).dt.days

df_Italy['Confirmed Change'] = df_Italy['Confirmed'].diff()

df_Italy.tail()
# Counting days from first confirmed case in each country (normalize)

initialDate_Spain = df_Spain[df_Spain['Confirmed'] >= 1]

initialDate_Spain = initialDate_Spain['Date'].min()

print(initialDate_Spain)

df_Spain = df_Spain[:]

df_Spain['Day Number'] = (df_Spain['Date'] - initialDate_Spain).dt.days

df_Spain['Confirmed Change'] = df_Spain['Confirmed'].diff()

df_Spain.tail()
# Counting days from first confirmed case in each country (normalize)

initialDate_SouthKorea = df_SouthKorea[df_SouthKorea['Confirmed'] >= 1]

initialDate_SouthKorea = initialDate_SouthKorea['Date'].min()

print(initialDate_SouthKorea)

df_SouthKorea = df_SouthKorea[:]

df_SouthKorea['Day Number'] = (df_SouthKorea['Date'] - initialDate_SouthKorea).dt.days

df_SouthKorea['Confirmed Change'] = df_SouthKorea['Confirmed'].diff()

df_SouthKorea.tail()
# Counting days from first confirmed case in each country (normalize)

initialDate_Mexico = df_Mexico[df_Mexico['Confirmed'] >= 1]

initialDate_Mexico = initialDate_Mexico['Date'].min()

print(initialDate_Mexico)

df_Mexico = df_Mexico[:]

df_Mexico['Day Number'] = (df_Mexico['Date'] - initialDate_Mexico).dt.days

df_Mexico['Confirmed Change'] = df_Mexico['Confirmed'].diff()

df_Mexico.tail()
# Comparison between Colombia and Canada behaviour

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.set_yscale('log')



plt.plot(df_Colombia['Day Number'], df_Colombia['Confirmed'], label='Colombia')

plt.plot(df_Canada['Day Number'], df_Canada['Confirmed'], label='Canada')

plt.plot(df_Italy['Day Number'], df_Italy['Confirmed'], label='Italy')

plt.plot(df_Spain['Day Number'], df_Spain['Confirmed'], label='Spain')

plt.plot(df_SouthKorea['Day Number'], df_SouthKorea['Confirmed'], label='South Korea')

plt.plot(df_Mexico['Day Number'], df_Mexico['Confirmed'], label='Mexico')



ax.legend()

ax.set_xlim(-1, 80)

ax.set_ylim(0, 1000000)

plt.xlabel('Days after first case')

plt.ylabel('log(#Cases)')



# Comparison between Colombia and Canada behaviour

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

ax.set_yscale('log')



plt.plot(df_Colombia['Day Number'], df_Colombia['Confirmed Change'],'*', label='Colombia')

plt.plot(df_Canada['Day Number'], df_Canada['Confirmed Change'],'*', label='Canada')

plt.plot(df_Italy['Day Number'], df_Italy['Confirmed Change'],'*', label='Italy')

#plt.plot(df_Spain['Day Number'], df_Spain['Confirmed Change'],'*', label='Spain')

#plt.plot(df_SouthKorea['Day Number'], df_SouthKorea['Confirmed Change'],'*', label='South Korea')

#plt.plot(df_Mexico['Day Number'], df_Mexico['Confirmed Change'],'*', label='Mexico')



ax.legend()

ax.set_xlim(-1, 80)

ax.set_ylim(0, 100000)

plt.xlabel('Days after first case')

plt.ylabel('#Confirmed cases change')
# Comparison between Colombia and Canada behaviour

fig, ax = plt.subplots(3,2, sharex=True, sharey=True, figsize=(8, 20))



         

plt.subplot(3, 2, 1)

plt.bar(df_Colombia['Day Number'], df_Colombia['Confirmed Change'], label='Colombia')

plt.title('Colombia')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 78, 0, 200])



plt.subplot(3, 2, 2)

plt.bar(df_Canada['Day Number'], df_Canada['Confirmed Change'], label='Canada')

plt.title('Canada')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 80, 0, 3000])



plt.subplot(3, 2, 3)

plt.bar(df_Italy['Day Number'], df_Italy['Confirmed Change'], label='Italy')

plt.title('Italy')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 80, 0, 8000])



plt.subplot(3, 2, 4)

plt.bar(df_Spain['Day Number'], df_Spain['Confirmed Change'], label='Spain')

plt.title('Spain')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 80, 0, 10000])



plt.subplot(3, 2, 5)

plt.bar(df_SouthKorea['Day Number'], df_SouthKorea['Confirmed Change'], label='South Korea')

plt.title('South Korea')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 80, 0, 1000])



plt.subplot(3, 2, 6)

plt.bar(df_Mexico['Day Number'], df_Mexico['Confirmed Change'], label='Mexico')

plt.title('Mexico')

plt.xlabel("Days after first case")

# xmin, xmax, ymin, ymax = plt.axis([-1, 80, 0, 500])



title_Fig = 'Change per day in number of cases'

fig.suptitle(title_Fig)



fig.text(0.04, 0.5, ('Number of cases'), va='center', rotation='vertical')
plt.plot(df_colombia.index, df_colombia['confirmed'] )

plt.plot(df_Colombia['Day Number'], df_Colombia['Confirmed'], label='Colombia')