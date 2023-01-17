%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Set some Pandas options

pd.set_option('max_columns', 30)

pd.set_option('max_rows', 20)



# Read the horse run data

df_runs = pd.read_csv('../input/runs.csv')

df_runs.head()
# We'll also need to get the race data

df_races = pd.read_csv('../input/races.csv', parse_dates=['date']).set_index('race_id')

df_races.head()
# Group horse runs by horse

df_horse_runs = df_runs.groupby('horse_id')
# Find the number of previous runs for a horse

def number_of_previous(horse_id, race_date):

    this_horse_runs = df_horse_runs.get_group(horse_id)

    return len(this_horse_runs[this_horse_runs['date'] < race_date])
df_runs['date'] = df_runs.apply(lambda run: df_races.loc[run['race_id'], 'date'], axis=1)

df_runs['no_previous'] = df_runs.apply(lambda run: number_of_previous(run['horse_id'], run['date']), axis=1)

df_runs[['race_id', 'date', 'horse_no', 'horse_id', 'no_previous']].iloc[1000:1005]
runs_vs_wins = df_runs.groupby('no_previous')['won'].sum()

runs_vs_wins.plot()
experience = df_runs.groupby('no_previous').size()

experience.plot()
strike_rate_vs_wins = df_runs.groupby('no_previous')['won'].mean()

strike_rate_vs_wins.plot()