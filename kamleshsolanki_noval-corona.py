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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
recoverd_data     = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

deaths_data       = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

confirmed_data    = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

COVID19_open_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')

COVID19_line_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')

covid_19_data     = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

nCOV_data         = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv',parse_dates=['Date','Last Update'])
nCOV_data.head()
nCOV_data.head(5)
pd.DataFrame({'Country':nCOV_data['Country'].unique()})
df = nCOV_data.groupby('Country').agg({'Confirmed':'sum'}).reset_index()
df
nCOV_data['Country'] = nCOV_data['Country'].replace('Mainland China','China')
df = nCOV_data.groupby('Country').agg({'Confirmed':'sum'}).reset_index()

plt.figure(figsize=(20,5))

fig = sns.barplot(data=df.sort_values('Confirmed', ascending=False), x='Country', y='Confirmed')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
plt.figure(figsize=(20,5))

fig = sns.barplot(data=df[df.Country != 'China'].sort_values('Confirmed', ascending=False), x='Country', y='Confirmed')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
df = nCOV_data.groupby('Country').agg({'Deaths':'sum'}).reset_index()

plt.figure(figsize=(20,5))

fig = sns.barplot(data=df.sort_values('Deaths', ascending=False), x='Country', y='Deaths')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
plt.figure(figsize=(20,5))

fig = sns.barplot(data=df[df.Country != 'China'].sort_values('Deaths', ascending=False), x='Country', y='Deaths')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
df = nCOV_data.groupby('Country').agg({'Recovered':'sum'}).reset_index()



plt.figure(figsize=(20,5))

fig = sns.barplot(data=df.sort_values('Recovered', ascending=False), x='Country', y='Recovered')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
df = nCOV_data.groupby('Country').agg({'Recovered':'sum'}).reset_index()



plt.figure(figsize=(20,5))

fig = sns.barplot(data=df[df.Country != 'China'].sort_values('Recovered', ascending=False), x='Country', y='Recovered')

_,labels = plt.xticks()

_ = fig.set_xticklabels(labels,rotation=30)
total    = nCOV_data['Confirmed'].sum() 

recoverd = nCOV_data['Recovered'].sum()

deaths   = nCOV_data['Deaths'].sum()

print('total confirmed case is {}'.format(total))

print('total Recoverd case is {}'.format(recoverd))

print('total Deaths case is {}'.format(deaths))
mortality_rate = deaths/total * 100

print('mortality rate to virus {}%'.format(mortality_rate))
recovery_rate = recoverd/total * 100

print('recovery rate to virus {}%'.format(recovery_rate))
df = pd.DataFrame([total,recoverd,deaths]).T

df.columns = ['Total','Recoverd','Deaths']

df.index = ['Cases']
df.plot(kind='bar')
nCOV_data['Date'].dt.year.unique()
nCOV_data['year'] = nCOV_data['Date'].dt.year

nCOV_data['month'] = nCOV_data['Date'].dt.month

nCOV_data['day'] = nCOV_data['Date'].dt.day
nCOV_data.groupby(['month']).agg({'Confirmed':'sum'})
nCOV_data.groupby(['day','month']).agg({'Confirmed':'sum'})