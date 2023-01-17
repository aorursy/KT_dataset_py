import pandas as pd

import numpy as np



g_mob = pd.read_csv('/kaggle/input/covid19-mobility-data/Global_Mobility_Report.csv', low_memory=False)

g_mob.columns = ['code', 'country', 's1', 's2', 'r1', 'r2', 'date', 'rec', 'groc', 'parks', 'transit', 'work', 'home']

g_mob = g_mob[pd.isnull(g_mob['s1'])].reset_index()

g_mob = g_mob.drop(labels=['index', 's1', 's2'], axis=1)

g_mob['date'] = pd.to_datetime(g_mob['date'])

start_date = '2020-03-01'

end_date = '2020-04-30'

mask = (g_mob['date'] >= start_date) & (g_mob['date'] <= end_date)

g_mob = g_mob.loc[mask].reset_index()

g_mob = g_mob.drop(labels=['index', 'code', 'r1', 'r2', 'date'], axis=1)

print(g_mob.head())
agg_gmob = g_mob.groupby('country').mean().reset_index()

print(agg_gmob.head())
covid = pd.read_csv('/kaggle/input/coronavirus-daily-stats-by-country/covid_world_stats.csv')

covid = covid[covid['Date']=='2020-05-15'].reset_index()

covid.rename(columns={'Country/Region': 'country', 

                     'Overall Deaths': 'deaths'}, inplace=True)

covid = covid[['country', 'deaths']]

covid['country'] = covid['country'].replace({'US':'United States'})

print(covid.head())
pop = pd.read_csv('/kaggle/input/population-by-country-2020/population_by_country_2020.csv')

pop.rename(columns={'Country (or dependency)': 'country',

                   'Population (2020)': 'population'}, inplace=True)

pop = pop[['country', 'population']]

print(pop.head())
covidpop = covid.merge(pop, how='inner')

covidpop['deaths_per_cap'] = (covidpop['deaths'] / covidpop['population']) * 100000

print(covidpop.head())



covidpm = covidpop.merge(agg_gmob, how='inner')

covidpm = covidpm[['country', 'deaths_per_cap', 'rec', 'groc', 'parks', 'transit', 'work', 'home']]

covidpm.rename(columns={'rec': 'Retail and Recreation',

                       'groc': 'Grocery and Pharmacy',

                       'parks': 'Parks',

                       'transit': 'Public Transport',

                       'work': 'Workplaces',

                       'home': 'Residential'}, inplace=True)

covidpm = covidpm.melt(id_vars=['country', 'deaths_per_cap'],

                      var_name='type',

                      value_name='change')

print(covidpm.head())



covidpm.to_csv('covidmobilitydeaths.csv', index=False)