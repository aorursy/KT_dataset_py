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

df_races = df_races[(df_races['race_class'] >= 1) & (df_races['race_class'] <= 5)].iloc[1000:2000]

df_races.head()
# We'll also need to get the horse run data for the above races

df_runs = pd.read_csv('../input/runs.csv')

df_runs = df_runs[df_runs['race_id'].isin(df_races.index.values)]

df_runs.head()
weight_vs_wins = df_runs.groupby('actual_weight')['won'].mean()

weight_vs_wins.plot();
df_runs['race_class'] = df_runs.apply(lambda run: df_races.loc[run['race_id'], 'race_class'], axis=1)

df_class_runs = df_runs.groupby('race_class')

df_class_runs['actual_weight'].mean()
# Add an 'actual_weight_mean' field to each run

df_mean_actual_weight = df_runs.groupby('race_id')['actual_weight'].mean().to_frame()

df_runs = df_runs.join(other=df_mean_actual_weight, on='race_id', how='outer', rsuffix='_mean')



# Add an 'actual_weight_var' field so we can see the relative weight carried by this horse

df_runs['actual_weight_var'] = df_runs.apply(lambda run: run['actual_weight'] - run['actual_weight_mean'], axis=1)
# Plot actual weight variance against strike rate

weight_var_vs_wins = df_runs.groupby('actual_weight_var')['won'].mean()

weight_var_vs_wins.plot();
df_runs['weight_ratio'] = df_runs['actual_weight'] / df_runs['declared_weight']

weight_ratio_vs_wins = df_runs.groupby('weight_ratio')['won'].mean()

weight_ratio_vs_wins.plot();
df_runs['distance'] = df_runs.apply(lambda run: df_races.loc[run['race_id']]['distance'], axis=1)

df_runs['weight_distance'] = df_runs['actual_weight'] * df_runs['distance']

weight_distance_vs_wins = df_runs.groupby('weight_distance')['won'].mean()

weight_distance_vs_wins.plot();