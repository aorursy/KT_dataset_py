import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# read tables from csv



df_contestants = pd.read_csv('../input/bachelorette-contestants.csv')

df_bachelorettes = pd.read_csv('../input/bachelorettes.csv')

df_all = df_contestants.merge(df_bachelorettes, on='Season', suffixes=('Contestant', 'Bachelorette'))
# find fraction of rose ceremonies passed for each contestant



weeks_in_season = dict(df_contestants.groupby('Season')['ElimWeek'].aggregate(np.max))



def progress(a, b):

    if np.isnan(a):

        return 1

    else:

        return (a-1) / weeks_in_season[b]

    

df_contestants['Progress'] = df_contestants[['ElimWeek', 'Season']].apply(

    lambda x: progress(x[0], x[1]), axis=1)
# find age differences between contestants and bachelorettes



df_contestants['AgeDifference'] = df_all['AgeContestant'] - df_all['AgeBachelorette']
df_contestants.plot(kind='scatter', marker='x', x='Progress', y='AgeDifference')