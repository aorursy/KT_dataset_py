import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Loading the data, parsing the dates & making date column as index column

df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv', parse_dates=['date'], index_col='date')

df.head()
# Checking for null values

df.info()
def human_format(num):

    num = float('{:.3g}'.format(num))

    magnitude = 0

    while abs(num) >= 1000:

        magnitude += 1

        num /= 1000.0

    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
# Getting uniuqe states from state column 

unique_states = df.state.unique()



print(unique_states)



print()



# Number of unique states

print(f'Number of unique states: {len(unique_states)}')
# Grouping data on the basis of states

states_group = df.groupby(['state'])
# Iterating through first item in states_group

for state, state_df in states_group:

    print(state)

    print(state_df)

    break
# Structure of states_data dictornary

# states_data = {

#     'state_name': {

#         'cases': 'total_num_of_cases',

#         'deaths': 'total_num_of_deaths'

#     }

# }



states_data = {}

for state, state_df in states_group:

    states_data[state] = {

        'cases': state_df.cases.sum(),

        'deaths': state_df.deaths.sum()

    }
states_df = pd.DataFrame(states_data).transpose()

states_df['state'] = states_df.index

states_df.head()
f, ax = plt.subplots(figsize=(10, 15))



sns.set_color_codes("pastel")

sns.barplot(x='cases', y='state', data=states_df, label="Total Cases", color="b")



sns.set_color_codes("muted")

sns.barplot(x='deaths', y='state', data=states_df, label="Total Deaths", color="b")



ax.set(ylabel="States", xlabel="Cases & Deaths in states")
results = states_df.sort_values('cases', ascending=False).head(5)



print('*** Top 5 states on the basis of number of cases *** \n')

for state in results.index:

    print('-'*20)

    print(f'{state} | {human_format(results.loc[state].cases)}')
results = states_df.sort_values('cases', ascending=True).head(5)



print('*** Bottom 5 states on the basis of number of cases *** \n')

for state in results.index:

    print('-'*20)

    print(f'{state} | {human_format(results.loc[state].cases)}')
results = states_df.sort_values('deaths', ascending=False).head(5)



print('*** Top 5 states on the basis of number of deaths *** \n')

for state in results.index:

    print('-'*20)

    print(f'{state} | {human_format(results.loc[state].deaths)}')
results = states_df.sort_values('deaths', ascending=True).head(5)



print('*** Bottom 5 states on the basis of number of deaths *** \n')

for state in results.index:

    print('-'*20)

    print(f'{state} | {human_format(results.loc[state].deaths)}')
states_df[states_df.cases == states_df.cases.max()]
states_df[states_df.deaths == states_df.deaths.max()]
ratio = round(states_df.deaths.sum() / states_df.cases.sum() * 100, 3)

f'Deaths to Cases ratio: {ratio}%'
# Getting New York state dataframe from states_group

for state, state_df in states_group:

    if state == 'New York':

        ny_df = pd.DataFrame(state_df)

        

ny_df.head()
# Grouping ny_df on the basis of date

ny_grp_by_dates = ny_df.groupby(['date'])
# Structure of ny_data dictornary

# ny_data = {

#     'date': {

#         'county': 'county_name',

#         'cases': 'total_num_of_cases_on_that_date'

#         'deaths': 'total_num_of_deaths_on_that_date'

#     }

# }



ny_data = {}

for date, date_df in ny_grp_by_dates:

    ny_data[date] = {

        'county': date_df.county.unique()[0],

        'cases': date_df.cases.sum(),

        'deaths': date_df.deaths.sum()

    }
ny_df = pd.DataFrame(ny_data).transpose()

ny_df['date'] = ny_df.index

ny_df.head()
ny_df.tail()
# Converting cases & deaths dtype to int for plotting charts

ny_df.cases = ny_df.cases.astype(int)

ny_df.deaths = ny_df.deaths.astype(int)
fig = plt.figure(figsize=(12, 4))



# Resampling cases on 'Weekly' frequency & then taking its average

ny_df.cases.resample('W').mean().plot(colormap=plt.cm.RdYlGn, grid=False)

plt.xlabel('Average of weekly cases in New York')

plt.ylabel('Cases')
fig = plt.figure(figsize=(12, 4))



# Resampling deaths on 'Weekly' frequency & then taking its average

ny_df.deaths.resample('W').mean().plot(colormap=plt.cm.cividis, grid=False)

plt.xlabel('Average of weekly deaths in New York')

plt.ylabel('Deaths')
ratio = round(ny_df.deaths.sum() / ny_df.cases.sum() * 100, 3)

f'Deaths to Cases ratio: {ratio}%'
# Grouping on the basis of counties

counties_group = df.groupby(['county'])
# Iterating through first item in counties_group

for county, county_df in counties_group:

    print(county)

    print(county_df)

    break
# Structure of county_data

# county_data = {

#     'county_name': {

#         'cases': 'total_cases_in_that_county',

#         'deaths': 'total_deaths_in_that_county'

#     }

# }



county_data = {}

for county, county_df in counties_group:

    county_data[county] = {

        'cases': county_df.cases.sum(),

        'deaths': county_df.deaths.sum()

    }
counties_df = pd.DataFrame(county_data).transpose()

counties_df['county'] = counties_df.index

counties_df.head()
results = counties_df.sort_values('cases', ascending=False).head(5)



print('*** Top 5 counties on the basis of number of cases *** \n')

for county in results.index:

    print('-'*20)

    print(f'{county} | {human_format(results.loc[county].cases)}')
results = counties_df.sort_values('cases', ascending=True).head(5)



print('*** Bottom 5 counties on the basis of number of cases *** \n')

for county in results.index:

    print('-'*20)

    print(f'{county} | {human_format(results.loc[county].cases)}')
results = counties_df.sort_values('deaths', ascending=False).head(5)



print('*** Top 5 counties on the basis of number of deaths *** \n')

for county in results.index:

    print('-'*20)

    print(f'{county} | {human_format(results.loc[county].deaths)}')
results = counties_df.sort_values('cases', ascending=True).head(5)



print('*** Bottom 5 counties on the basis of number of deaths *** \n')

for county in results.index:

    print('-'*20)

    print(f'{county} | {human_format(results.loc[county].deaths)}')
# Getting ny_city_df from counties_group

for county, county_df in counties_group:

    if county == 'New York City':

        ny_city_df = county_df

        

ny_city_df.head()
ny_city_df.tail()
fig = plt.figure(figsize=(12, 4))



# Resampling cases on 'Weekly' frequency & then taking its average

ny_city_df.cases.resample('W').mean().plot(colormap=plt.cm.RdYlGn, grid=False)

plt.xlabel('Average of weekly cases in New York City')

plt.ylabel('Cases')
fig = plt.figure(figsize=(12, 4))



# Resampling deaths on 'Weekly' frequency & then taking its average

ny_city_df.deaths.resample('W').mean().plot(colormap=plt.cm.cividis, grid=False)

plt.xlabel('Average of weekly cases in New York City')

plt.ylabel('Cases')
ratio = round(ny_city_df.deaths.sum() / ny_city_df.cases.sum() * 100, 3)

f'Deaths to Cases ratio: {ratio}%'