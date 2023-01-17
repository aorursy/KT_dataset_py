import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import matplotlib.pyplot as plt

%matplotlib inline
# inspired by https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons

confirmed0 = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

print(confirmed0.shape)

confirmed0.head()
# rename time series columns to the confirmed_YYYY-MM-DD format

cols_confirmed0 = list(confirmed0.columns)

cols_confirmed1 = cols_confirmed0

cols_confirmed1[4:] = [f"confirmed_{datetime.datetime.strptime(d, '%m/%d/%y').isoformat()[:10]}" for d in cols_confirmed0[4:]]

confirmed1 = confirmed0.copy()

confirmed1.columns = cols_confirmed1

confirmed1.columns[:10]
confirmed_last = confirmed1.columns[-1]

confirmed_last
# see the biggest countries in terms of number of confirmed

confirmed1.loc[:, ['Country/Region', confirmed_last]].sort_values(confirmed_last, ascending=False).head(30).reset_index()
# total number of confirmed worldwide

confirmed1[confirmed_last].sum()
# What the Provice/State is here for?

confirmed1.loc[confirmed1['Province/State'].notnull(), ['Country/Region', 'Province/State', confirmed_last]]
# Which countries are split by province?

tmp = confirmed1.loc[confirmed1['Province/State'].notnull(), ['Country/Region', 'Province/State', confirmed_last]]

tmp.groupby('Country/Region').agg([len, sum])
ind = confirmed1['Country/Region'].isin(['Denmark', 'France', 'Netherlands', 'United Kingdom']) & confirmed1['Province/State'].isnull()

confirmed1.loc[ind, ['Country/Region', 'Province/State', confirmed_last]]
# what about Taiwan?

confirmed1.loc[confirmed1['Country/Region'].str.startswith('Tai'), ['Country/Region', 'Province/State', confirmed_last]]
deaths0 = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

print(deaths0.shape)

deaths0.head()
# rename time series columns to the deaths_YYYY-MM-DD format

cols_deaths0 = list(deaths0.columns)

cols_deaths1 = cols_deaths0

cols_deaths1[4:] = [f"deaths_{datetime.datetime.strptime(d, '%m/%d/%y').isoformat()[:10]}" for d in cols_deaths0[4:]]

deaths1 = deaths0.copy()

deaths1.columns = cols_deaths1

deaths1.columns[:10]
deaths_last = deaths1.columns[-1]

deaths_last
# total number of deaths worldwide

deaths1[deaths_last].sum()
popurl = "/kaggle/input/population/population_csv.txt"

pop0 = pd.read_csv(popurl, sep=",")

pop0.head()
(pop0['Year'].min(), pop0['Year'].max())
pop1 = pop0.loc[pop0['Year']==2016, :]

pop1.head()
# The World population is 7e9, here the sum does not match. Some regions are counted multiple times

pop1.Value.sum()/1e9
# Clearly this is not MECE. But this is not a problem, because we will merge by country name and those regions will drop out.

pop1.sort_values(by='Value', ascending=False).head(20)
# from https://en.wikipedia.org/wiki/List_of_Chinese_administrative_divisions_by_population

chinaurl = "/kaggle/input/population-china-v2/population_china_2017.csv"

china0 = pd.read_csv(chinaurl, sep="\t")

print(china0.population.sum()/1e9)

china0
# prepare china for appending to the population dataset

china1 = china0.copy()

china1['Country/Region'] = 'China'

china1['country_code'] = 'CHN'

china1['year'] = 2017

china1.at[33, 'Country/Region'] = 'Taiwan*'

china1.at[33, 'Province/State'] = np.nan

china1
pop2 = pop1.copy()

pop2.columns = ['Country/Region', 'country_code', 'year', 'population']

chn = pop2.index[pop2['country_code'] == 'CHN']

print(chn)

pop2 = pop2.drop(chn).append(china1, sort=False).reset_index()

pop2.loc[pop2['country_code'] == 'CHN', :]
# list the top 20

pop2.sort_values('population', ascending=False).head(20)
dfc0 = confirmed1.merge(deaths1, how='left', on=['Country/Region', 'Province/State'])

print(confirmed1.shape)

print(deaths1.shape)

print(dfc0.shape)
pop3 = pop2.set_index('country_code')

pop3.at['CZE', 'Country/Region'] = 'Czechia'

pop3.at['SVK', 'Country/Region'] = 'Slovakia'

pop3.at['EGY', 'Country/Region'] = 'Egypt'

pop3.at['IRN', 'Country/Region'] = 'Iran'

pop3.at['RUS', 'Country/Region'] = 'Russia'

pop3.at['USA', 'Country/Region'] = 'US'

pop3.at['VEN', 'Country/Region'] = 'Venezuela'

pop3.at['KOR', 'Country/Region'] = 'Korea, South'

df = dfc0.merge(pop3, how='left', on=['Country/Region', 'Province/State']).reset_index()

print(df.population.sum()/1e9)  # should be 7.1

print(df.loc[df.population.isnull(), confirmed_last].sum())  # not matched confirmed cases => immaterial

print(df.loc[df.population.isnull(), ['Country/Region', 'Province/State', confirmed_last]].sort_values(confirmed_last, ascending=False).reset_index())  # list of not matched
# on the latest snapshot calculate the infect rate and mortality rate and show top 50

dff = df.loc[df.population.notnull(), :]

print(dff[confirmed_last].sum())

print(dff[deaths_last].sum())

print(dff.population.sum()/1e9)

df_latest = dff.loc[:, ['Country/Region', 'Province/State', confirmed_last, deaths_last, 'population']]

df_latest['infect_rate'] = df_latest[confirmed_last] / df_latest['population'] * 1000

df_latest['mortality_rate'] = df_latest[deaths_last] / df_latest['population'] * 100000

df_latest.sort_values('mortality_rate', ascending=False).head(50).reset_index()
# look at the china which provinces are worth to include into selection

df.loc[df['Country/Region']=='China', ['Country/Region', 'Province/State', confirmed_last, deaths_last, 'population']].sort_values(confirmed_last, ascending=False)



# We have finally kept just Hubei, because other provinces did not show to be useful in the end.
list1 = ['US', 'Czechia', 'Italy', 'Spain', 'Germany', 'United Kingdom', 'Iran', 'Korea, South', 'Austria', 'France', 'Norway', 'Sweden', 'Switzerland']

filt1 = df['Country/Region'].isin(list1) & df['Province/State'].isnull()

filt2 = (df['Country/Region'] == 'China') & df['Province/State'].isin(['Hubei'])

filt = filt1 | filt2

print(np.sum(filt))

df.loc[filt, ['Country/Region', 'Province/State', 'confirmed_2020-03-31', 'deaths_2020-03-31', 'population']]
# how large part of the world the selection covers?

def part(q, f):

    p = q[f].sum() / q.sum()

    print(f"{q.name}: {100*p:.1f}%")



part(df['population'], filt)

part(df[confirmed_last], filt)

part(df[deaths_last], filt)
# create names to use later in the legend

names = [country if type(province).__name__=='float' else f"{country}/{province}" for country, province in zip(df.loc[filt, 'Country/Region'], df.loc[filt, 'Province/State'])]

names
col_confirmed = df.columns.str.startswith('confirmed_')

col_deaths = df.columns.str.startswith('deaths_')

confirmed = df.loc[filt, col_confirmed].values

new_confirmed = np.diff(confirmed, axis=1, prepend=0)

deaths = df.loc[filt, col_deaths].values

population = df.population[filt].values

infect_rate = confirmed / population[:,np.newaxis] * 1000

new_infect_rate = new_confirmed / population[:,np.newaxis] * 1000

mortality_rate = deaths / population[:,np.newaxis] * 100000

lethality_rate = deaths / confirmed * 1000
# prepare also time axis for the plots

tm_dt = [datetime.datetime.strptime(d, '%m/%d/%y') for d in confirmed0.columns[4:]]

tm = np.array([datetime.date(d.year, d.month, d.day) for d in tm_dt])

print(len(tm))

tm
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, infect_rate[i, :], lw=lw)

plt.ylabel("Infect rate", fontsize=18)

plt.title("Number of confirmed cases per 1000 inhabitants", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, infect_rate[i, :], lw=lw)

plt.ylabel("Infect rate (log scale)", fontsize=18)

plt.yscale("log")

plt.ylim([1e-2, 4])

plt.title("Number of confirmed cases per 1000 inhabitants", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, new_infect_rate[i, :], lw=lw)

plt.ylabel("Infect rate growth", fontsize=18)

plt.title("Daily increase in number of confirmed cases per 1000 inhabitants", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, lethality_rate[i, :], lw=lw)

plt.ylim([0, 150])

plt.ylabel("Lethality rate", fontsize=18)

plt.title("Number of deaths per 1000 confirmed cases", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, mortality_rate[i, :], lw=lw)

plt.ylim([0, 20])

plt.ylabel("Mortality rate", fontsize=18)

plt.title("Number of deaths per 100K inhabitants", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)
f = plt.figure(figsize=(20,12))

f.add_subplot(111)

for i, name in enumerate(names):

    lw = 5 if name == 'Czechia' else 2

    plt.plot(tm, mortality_rate[i, :], lw=lw)

plt.yscale("log")

plt.ylim([1e-2, 40])

plt.ylabel("Mortality rate (log scale)", fontsize=18)

plt.title("Number of deaths per 100K inhabitants", fontsize=18)

plt.legend(names)

plt.grid(alpha=0.3)