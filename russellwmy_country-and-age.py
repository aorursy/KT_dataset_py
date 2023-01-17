%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# Set some Pandas options

pd.set_option('max_columns', 30)

pd.set_option('max_rows', 20)



# Read the race data

df_races = pd.read_csv('../input/races.csv', parse_dates=['date']).set_index('race_id')



# To save some time with feature calculations, just use a subset of 1000 handicap races

# df_races = df_races[(df_races['race_class'] >= 1) & (df_races['race_class'] <= 5)].iloc[1000:2000]

df_races.head()
# We'll also need to get the horse run data for the above races

df_runs = pd.read_csv('../input/runs.csv')

df_runs = df_runs[df_runs['race_id'].isin(df_races.index.values)]

df_runs.head()
result = df_runs.groupby([ 'horse_country'])['won'].mean()

result.plot.bar();
result = df_runs[df_runs.horse_country.isin(['AUS', 'NZ', 'JPN'])].groupby([ 'horse_age'])['won'].mean()

result.plot();
result = df_runs[df_runs.horse_country.isin(['GB', 'USA'])].groupby([ 'horse_age'])['won'].mean()

result.plot();