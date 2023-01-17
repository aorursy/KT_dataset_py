import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os



pd.set_option('display.max_rows', 500)

   

input_path = '/kaggle/input/novel-corona-virus-2019-dataset'

N_COUNTRIES = 20
df = pd.read_csv(f'{input_path}/covid_19_data.csv')

print(f'Number of entries: {df.shape[0]}')
country_df = df.copy()

country_df['Province/State'] = country_df['Province/State'].fillna('No provinces')

country_df = country_df.groupby(['Province/State', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered'].max()

country_df = country_df.groupby('Country/Region').sum()



country_df = pd.DataFrame(country_df)

country_df = country_df.sort_values('Deaths', ascending=False).iloc[:N_COUNTRIES]
country_df
countries_per_day = df.copy()

countries_per_day['Province/State'] = countries_per_day['Province/State'].fillna('No provinces')

countries_per_day = pd.DataFrame(countries_per_day.groupby(['ObservationDate', 'Country/Region'])['Deaths'].sum()).reset_index()
after_first_death = countries_per_day.copy()

after_first_death = after_first_death[after_first_death['Deaths'] > 0]



comparison_df = pd.DataFrame()

countries_to_compare = ['Brazil', 'Italy', 'Spain', 'US']



for country in countries_to_compare:

    temp = after_first_death[after_first_death['Country/Region'] == country]

    

    for column in list(temp.columns):

        if column == 'Country/Region': 

            continue

        

        pad_size = 70 - len(temp[f'{column}'])

        comparison_df[f'{country}{column}'] = np.pad(temp[f'{column}'].values, (0, pad_size))



comparison_df = comparison_df.replace(0, np.nan).dropna(how='all')

comparison_df = comparison_df.dropna(thresh=5)
comparison_df.replace(np.nan, '')
comparison_df.replace(np.nan, '')
fig = plt.figure(figsize=(14,9))



for country in countries_to_compare:

    plt.plot(comparison_df.index, f'{country}Deaths', data=comparison_df, marker='o')



plt.legend()
comparison_df[['BrazilDeaths', 'ItalyDeaths', 'SpainDeaths', 'USDeaths']] = comparison_df[['BrazilDeaths', 'ItalyDeaths', 'SpainDeaths', 'USDeaths']].diff()

comparison_df = comparison_df.dropna(thresh=5)
fig = plt.figure(figsize=(14,9))



for country in countries_to_compare:

    plt.plot(comparison_df.index, f'{country}Deaths', data=comparison_df, marker='o')



plt.legend()