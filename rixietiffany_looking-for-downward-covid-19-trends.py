import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# Clean Data

def clean_df(df):

    df = df.drop(['Lat', 'Long', 'Province/State'], axis=1)

    df = df.groupby('Country/Region').sum()

    df = df.stack()

    df = df.reset_index()

    df.columns = ['Country/Region', 'Date', 'Cases']

    df['Date'] =  pd.to_datetime(df['Date'], format='%m/%d/%y')

    df = df.set_index(['Country/Region', 'Date',])

    return df
# Confirmed Cases

confirmed_df = clean_df(pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv'))

confirmed_df
# Recovered Cases

recovered_df = clean_df(pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv'))

recovered_df
# Deaths

deaths_df = clean_df(pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv'))

deaths_df
# Active Cases

active_df = confirmed_df - recovered_df - deaths_df

active_df
# Combine All into One Table

combined_df = active_df.copy()

combined_df.columns = ['Active Cases']

combined_df['Total Cases'] = confirmed_df['Cases']

combined_df['Deaths'] = deaths_df['Cases']

combined_df['Recovered'] = deaths_df['Cases']

combined_df
today = combined_df.index.get_level_values(1).max()
# Add Centered Moving Average (CMA) Change (window = 1 week)

cma_change = combined_df.groupby(level=0).diff().rolling(7, center=True).mean().add_suffix(' CMA Change')

combined_df = pd.concat([combined_df, cma_change], axis=1)

combined_df
# Total Cases CMA Change / Total Cases

combined_df['% Change in Total Cases'] = combined_df['Total Cases CMA Change'] / combined_df['Total Cases']

combined_df
# Total Cases CMA Change / Active Cases

combined_df['% Change in Total Cases Over Active Cases'] = combined_df['Total Cases CMA Change'] / combined_df['Active Cases']

combined_df
# Lowest Total Cases CMA Percentage Change for Recent Data

recent_data = combined_df[combined_df.index.get_level_values(1) == today - timedelta(3)]

lowest_percentage_change = recent_data[recent_data['Total Cases'] > 100].sort_values(by='% Change in Total Cases').reset_index()['Country/Region']

lowest_percentage_change[:10]
# Countries which Have Ever Had Negative Active Cases Percentage Change

negative_active_cases_change = combined_df[(combined_df['Total Cases'] > 100) & (combined_df['Active Cases CMA Change'] < 0)].sort_values(by='Active Cases CMA Change').reset_index()['Country/Region'].unique()

negative_active_cases_change
# Countries with the Most Total Cases

temp = combined_df.reset_index()

top_countries = list(temp[temp['Date'] == today].sort_values(by='Total Cases', ascending=False)['Country/Region'])
# Graph of Active Cases over Time of the 10 Countries with the Highest Total Cases

plt.figure(figsize=[20, 10])



temp = combined_df.reset_index()

sns.lineplot(x='Date', y='Active Cases', hue='Country/Region', data=temp[temp['Country/Region'].isin(top_countries[:10])])
# Graph of Active Cases over Time of the 10 Countries with the Lowest % Total Cases Change (over Total Cases)

plt.figure(figsize=[20, 10])



temp = combined_df.reset_index()

sns.lineplot(x='Date', y='Active Cases', hue='Country/Region', data=temp[temp['Country/Region'].isin(lowest_percentage_change[:10])])
# Graph of Active Cases over Time of the Countries Which Have Ever Had a Negative Active Cases CMA Change

plt.figure(figsize=[20, 10])



temp = combined_df.reset_index()

sns.lineplot(x='Date', y='Active Cases', hue='Country/Region', data=temp[temp['Country/Region'].isin(negative_active_cases_change)])
# Log Graph of Total Cases CMA Increase vs Total Cases the 10 Countries with the Highest Total Cases

plt.figure(figsize=[20, 10])



plt.xscale('log')

plt.yscale('log')



temp = combined_df.reset_index()

sns.lineplot(x='Total Cases', y='Total Cases CMA Change', hue='Country/Region', data=temp[temp['Country/Region'].isin(top_countries[:10])], ci=None)

plt.plot([0,400000],[0,400000], color='black')
# Log Graph of Total Cases CMA Increase vs Total Cases of the 10 Countries with the Lowest % Total Cases Change (over Total Cases)

plt.figure(figsize=[20, 10])



plt.xscale('log')

plt.yscale('log')



temp = combined_df.reset_index()



sns.lineplot(x='Total Cases', y='Total Cases CMA Change', hue='Country/Region', data=temp[temp['Country/Region'].isin(lowest_percentage_change[:10])], ci=None)

plt.plot([0,400000],[0,400000], color='black')
# Log Graph of Total Cases CMA Increase vs Total Cases of the Countries Which Have Ever Had a Negative Active Cases CMA Change

plt.figure(figsize=[20, 10])



plt.xscale('log')

plt.yscale('log')



temp = combined_df.reset_index()



sns.lineplot(x='Total Cases', y='Total Cases CMA Change', hue='Country/Region', data=temp[temp['Country/Region'].isin(negative_active_cases_change)], ci=None)

plt.plot([0,400000],[0,400000], color='black')