# standard imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# reading dataset

df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
# checking first 5 rows

df.head()
# checking the shape of the dataset

print(f"{df.shape[0]} rows & {df.shape[1]} columns")
# checking min/max/avg stats

df.describe()
# checking entries/counts/datatypes

df.info()

# checking unique state counts

print(f"{df['state'].nunique()} states in this dataset")
# states sorted by number of deaths

deaths_by_state = df.sort_values('deaths', ascending=False)

state_deaths = deaths_by_state.drop_duplicates(subset="state", keep='first', inplace=True) 

state_deaths = deaths_by_state.nlargest(55, 'deaths').reset_index()



state_deaths
# top-ten deadliest states in the US

state_deaths = state_deaths[:10]

state_deaths
# visualization of top-ten deadliest states

plt.figure(figsize=(18,9))

chart = sns.barplot(state_deaths['deaths'], state_deaths['state'])

plt.xticks(horizontalalignment='center',fontsize='large')

plt.yticks(verticalalignment='center',fontsize='large')

plt.xlabel(xlabel='Deaths', fontsize='large')

plt.ylabel(ylabel='State', fontsize='large')

plt.title('Top 10 deadliest States', fontsize='x-large')
# checking unique county counts

print(f"{df['county'].nunique()} counties in this dataset")
# counties sorted by number of deaths

deaths_by_county = df.sort_values('deaths', ascending=False)

county_deaths = deaths_by_county.drop_duplicates(subset="county", keep='first', inplace=True) 

county_deaths = deaths_by_county.nlargest(1913, 'deaths').reset_index()



county_deaths
# top-ten deadliest counties in the US

county_deaths = county_deaths[:10]

county_deaths
# visualization of top-ten deadliest counties

plt.figure(figsize=(18,9))

chart = sns.barplot(county_deaths['deaths'], county_deaths['county'])

plt.xticks(horizontalalignment='center',fontsize='large')

plt.yticks(verticalalignment='center',fontsize='large')

plt.xlabel(xlabel='Deaths', fontsize='large')

plt.ylabel(ylabel='County', fontsize='large')

plt.title('Top 10 deadliest Counties', fontsize='x-large')
def main():

    '''

    outputs a visualization of top-ten deadliest counties by state

    '''

    for each in sorted(df['state'].unique()):

        state = df[df['state'] == each]

        state.sort_values('deaths', inplace=True, ascending=False)

        state.drop_duplicates(subset='county', keep='first', inplace=True)

        state = state[:10]

        plt.figure(figsize=(18,9))

        chart = sns.barplot(state['deaths'], state['county'])

        plt.xticks(horizontalalignment='center',fontsize='large')

        plt.yticks(verticalalignment='center',fontsize='large')

        plt.xlabel(xlabel='Deaths', fontsize='large')

        plt.ylabel(ylabel='County', fontsize='large')

        plt.title(f'{each} top 10 deadliest Counties', fontsize='x-large')

        yield



state = main()
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)
next(state)