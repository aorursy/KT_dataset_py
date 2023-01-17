import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os



from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
covid_19_data_complete = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
covid_19_data_complete.head(5)
df = covid_19_data_complete[covid_19_data_complete['Country/Region'] == 'Russia']
df.head(5)
df_confirmed = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)

df_confirmed['count'] = df['Confirmed']

df_confirmed['type'] = 'Confirmed'
df_confirmed.head(5)
df_deaths = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)

df_deaths['count'] = df['Deaths']

df_deaths['type'] = 'Deaths'
df_deaths.head(5)
df_recovered = df.drop(['Confirmed', 'Deaths', 'Recovered'], axis=1)

df_recovered['count'] = df['Recovered']

df_recovered['type'] = 'Recovered'
df_recovered.head(5)
frames = [df_confirmed, df_deaths, df_recovered]

df_for_eda = pd.concat(frames, sort=False)
df_for_eda['Date'] = pd.to_datetime(df_for_eda['Date'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.lineplot(x=df_for_eda['Date'], y=df_for_eda['count'], hue=df_for_eda['type'])

rotation = plt.setp(ax.get_xticklabels(), rotation=45)